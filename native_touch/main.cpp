#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iostream>
#include <io.h>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

static constexpr std::array<int,3> plan_fingers{3,4,5};
static constexpr auto contact_slots = plan_fingers.back()+1;
static constexpr int64_t plan_admission_us = 200000;
static constexpr int64_t board_plan_horizon_us = 1000000;

// The microsecond timestamps have QueryPerformanceCounter's shared system epoch.
static int64_t timestamp_us() {
    LARGE_INTEGER counter{}, frequency{};
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (counter.QuadPart / frequency.QuadPart) * 1000000LL +
           (counter.QuadPart % frequency.QuadPart) * 1000000LL / frequency.QuadPart;
}

// A slow or disconnected reader must never block the input/timing thread.
class Output {
public:
    Output() {
        if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_OUTPUT_HANDLE), GetCurrentProcess(),
                             &handle_, 0, FALSE, DUPLICATE_SAME_ACCESS))
            throw std::runtime_error("cannot duplicate protocol output pipe");
        std::thread([this] { write_loop(); }).detach();
    }
    void send(std::string line, bool telemetry = false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        if (queue_.size() >= 512) {
            if (telemetry) { ++dropped_; return; }
            auto it = std::find_if(queue_.begin(), queue_.end(), [](const Item& x) { return x.telemetry; });
            if (it != queue_.end()) queue_.erase(it);
            else queue_.pop_front();
            ++dropped_;
        }
        queue_.push_back({std::move(line) + "\n", telemetry});
        cv_.notify_one();
    }
    bool drain_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        return drained_.wait_for(lock, timeout, [this] { return queue_.empty() && !writing_; });
    }
private:
    struct Item { std::string line; bool telemetry; };
    HANDLE handle_;
    std::mutex mutex_;
    std::condition_variable cv_, drained_;
    std::deque<Item> queue_;
    bool writing_ = false, closed_ = false;
    uint64_t dropped_ = 0;
    void write_loop() {
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return !queue_.empty() || closed_; });
                if (closed_) return;
                line = std::move(queue_.front().line);
                queue_.pop_front();
                writing_ = true;
            }
            DWORD written = 0;
            const bool ok = WriteFile(handle_, line.data(), static_cast<DWORD>(line.size()), &written, nullptr) && written == line.size();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                writing_ = false;
                if (!ok) { closed_ = true; queue_.clear(); }
                drained_.notify_all();
                if (closed_) return;
            }
        }
    }
};

class MuMu {
public:
    explicit MuMu(const std::wstring& root, int index, bool mock) : mock_(mock) {
        if (mock_) {
            wchar_t value[16]{};
            if (GetEnvironmentVariableW(L"NINJA_TOUCH_MOCK_DELAY_MS", value, 16)) {
                try { mock_delay_ms_ = std::clamp(std::stoi(value), 0, 500); } catch (...) {}
            }
            return;
        }
        namespace fs = std::filesystem;
        fs::path base(root);
        fs::path sdk = base / L"nx_device/12.0/shell/sdk";
        if (!fs::exists(sdk / L"external_renderer_ipc.dll")) sdk = base / L"nx_main/sdk";
        if (!fs::exists(sdk / L"external_renderer_ipc.dll")) throw std::runtime_error("MuMu SDK DLL not found");
        SetDllDirectoryW(sdk.c_str());
        library_ = LoadLibraryW((sdk / L"external_renderer_ipc.dll").c_str());
        if (!library_) throw std::runtime_error("MuMu SDK DLL load failed");
        try {
            connect_ = symbol<Connect>("nemu_connect");
            disconnect_ = symbol<Disconnect>("nemu_disconnect");
            display_ = symbol<Display>("nemu_get_display_id");
            capture_ = symbol<Capture>("nemu_capture_display");
            down_ = symbol<Down>("nemu_input_event_finger_touch_down");
            up_ = symbol<Up>("nemu_input_event_finger_touch_up");
            handle_ = connect_(root.c_str(), index);
            if (handle_ <= 0) throw std::runtime_error("MuMu connection failed");
            screen_ = display_(handle_, "com.pandadastudio.ninjamustdie3.mi", 0);
            if (screen_ != 0) throw std::runtime_error("game display is unavailable");
            int width = 0, height = 0;
            if (capture_(handle_, 0, 0, &width, &height, nullptr) || width != 1280 || height != 720)
                throw std::runtime_error("game display must be 1280x720");
        } catch (...) {
            close();
            throw;
        }
    }
    ~MuMu() { close(); }
    int down(int id, int x, int y) {
        if (mock_) {
            if (mock_delay_ms_) std::this_thread::sleep_for(std::chrono::milliseconds(mock_delay_ms_));
            return 0;
        }
        return down_(handle_, screen_, id, x, y);
    }
    int up(int id) { return mock_ ? 0 : up_(handle_, screen_, id); }
private:
    using Connect = int (__cdecl *)(const wchar_t*, int);
    using Disconnect = void (__cdecl *)(int);
    using Display = int (__cdecl *)(int, const char*, int);
    using Capture = int (__cdecl *)(int, unsigned int, int, int*, int*, unsigned char*);
    using Down = int (__cdecl *)(int, int, int, int, int);
    using Up = int (__cdecl *)(int, int, int);
    template<class T> T symbol(const char* name) {
        auto value = GetProcAddress(library_, name);
        if (!value) throw std::runtime_error(std::string("missing SDK function ") + name);
        return reinterpret_cast<T>(value);
    }
    void close() {
        if (handle_ > 0 && disconnect_) disconnect_(handle_);
        handle_ = 0;
        if (library_) FreeLibrary(library_);
        library_ = nullptr;
    }
    bool mock_ = false;
    int mock_delay_ms_ = 0;
    HMODULE library_ = nullptr;
    int handle_ = 0, screen_ = 0;
    Connect connect_ = nullptr;
    Disconnect disconnect_ = nullptr;
    Display display_ = nullptr;
    Capture capture_ = nullptr;
    Down down_ = nullptr;
    Up up_ = nullptr;
};

struct Shared {
    std::mutex mutex;
    std::condition_variable cv;
    bool skating = false, quit = false, idle = true, fatal = false;
    bool announce_start = false;
    std::string start_id;
    uint64_t generation = 0;
    Clock::time_point deadline{};
    struct Tap {
        int x = 0, y = 0;
        Clock::time_point deadline{};
        std::string id;
        int hold_ms = 25;
        int64_t deadline_us = 0;
        bool glide = false;
        int64_t at_us = 0, revision = 0;
        int index = -1, finger = 3;
    };
    std::optional<Tap> tap;
    std::optional<Tap> skill;
    struct Plan {
        std::array<Tap,16> actions;
        size_t begin = 0, end = 0;
        int64_t revision = 0;
    };
    std::array<Plan,contact_slots> plans;
    int64_t plan_floor = 0;
};

static std::string tap_dropped(const Shared::Tap& tap) {
    if (tap.index >= 0) return "PLAN_DROPPED " + std::to_string(tap.finger) + " " +
        std::to_string(tap.revision) + " " + std::to_string(tap.index);
    return (tap.id.empty() ? "TAP_DROPPED " : "TAP_DROPPED2 " + tap.id + " ") +
           std::to_string(tap.x) + " " + std::to_string(tap.y);
}

static void clear_plans(Shared& state, Output& output, int64_t barrier = 0) {
    for (int id : plan_fingers) {
        auto& plan = state.plans[id];
        while (plan.begin < plan.end) output.send(tap_dropped(plan.actions[plan.begin++]));
        state.plan_floor = std::max(state.plan_floor,plan.revision);
    }
    state.plan_floor = std::max(state.plan_floor,barrier);
}

class Engine {
public:
    Engine(MuMu& device, Shared& state, Output& output, bool slide, bool jump_two) : device_(device), state_(state), output_(output), slide_(slide), jump_two_(jump_two) {}
    void run() {
        try { loop(); }
        catch (const std::exception& e) {
            const bool released = release_all();
            {
                std::lock_guard<std::mutex> lock(state_.mutex);
                state_.fatal = true;
                state_.skating = false;
                if (state_.tap) {
                    drop(*state_.tap);
                }
                state_.tap.reset();
                if (state_.skill) drop(*state_.skill);
                state_.skill.reset();
                clear_plans(state_,output_);
                state_.idle = released;
                state_.quit = true;
                state_.cv.notify_all();
            }
            output_.send(std::string("ERROR ") + e.what());
            if (!released) output_.send("ERROR touch_release_failed");
        }
    }
private:
    struct Step { int ms, id, x, y; bool down; };
    // Matches the independently recorded MuMu macro's ~108ms cycle.  The
    // game must still be checked visually; SDK success only means accepted.
    const Step steps_[7] = {{0,1,190,620,true}, {8,1,190,700,true},
        {8,2,1090,630,true}, {40,2,0,0,false}, {40,1,0,0,false},
        {60,1,190,620,true}, {72,1,190,700,true}};
    const Step jump_steps_[6] = {{0,2,1090,630,true}, {35,2,0,0,false},
        {160,2,1090,630,true}, {195,2,0,0,false},
        {320,1,190,620,true}, {328,1,190,700,true}};
    MuMu& device_;
    Shared& state_;
    Output& output_;
    bool slide_;
    bool jump_two_;
    bool held_[contact_slots]{};
    // Each finger consumes its own mailbox with an independent release time.
    // One timer loop advances all consumers without sleeping through a hold.
    struct FingerConsumer {
        std::optional<Shared::Tap> active;
        Clock::time_point release_at{};
    } consumers_[contact_slots];
    bool cycle_ = false;
    unsigned step_ = 0;
    uint64_t cycle_count_ = 0;
    Clock::time_point cycle_base_{};
    Clock::time_point next_cycle_{};
    Clock::time_point glide_until_{};
    int64_t cycle_start_us_ = 0;
    uint64_t observed_generation_ = 0;

    std::pair<int64_t,int64_t> touch_down(int id, int x, int y) {
        held_[id] = true; // A failed SDK call can leave an uncertain contact.
        const auto start = timestamp_us();
        int result = device_.down(id, x, y);
        const auto end = timestamp_us();
        output_.send("EVENT " + std::to_string(start) + " " + std::to_string(end) +
            " DOWN " + std::to_string(id) + " " + std::to_string(x) + " " + std::to_string(y) +
            " " + std::to_string(result), true);
        if (result) throw std::runtime_error("touch_down_failed");
        return {start,end};
    }
    void drop(const Shared::Tap& tap) {
        output_.send(tap_dropped(tap));
    }
    Clock::time_point next_release(Clock::time_point wake) const {
        for (int id : plan_fingers) {
            if (consumers_[id].active) wake = std::min(wake,consumers_[id].release_at);
            const auto& plan = state_.plans[id];
            if (!consumers_[id].active && plan.begin < plan.end) {
                const auto delay = std::max<int64_t>(0,plan.actions[plan.begin].at_us-timestamp_us());
                wake = std::min(wake,Clock::now()+std::chrono::microseconds(delay));
            }
        }
        return wake;
    }
    bool has_plans() const {
        return std::any_of(plan_fingers.begin(),plan_fingers.end(),[this](int id) {
            const auto& plan = state_.plans[id];
            return plan.begin < plan.end;
        });
    }
    void touch_up(int id) {
        if (!held_[id]) return;
        const auto start = timestamp_us();
        int result = device_.up(id);
        const auto end = timestamp_us();
        output_.send("EVENT " + std::to_string(start) + " " + std::to_string(end) +
            " UP " + std::to_string(id) + " " + std::to_string(result), true);
        if (result) throw std::runtime_error("touch_up_failed");
        held_[id] = false;
    }
    bool release_all() noexcept {
        bool okay = true;
        for (int id = contact_slots-1; id >= 1; --id) {
            try { touch_up(id); } catch (...) { okay = false; }
        }
        for (int id : plan_fingers) {
            if (consumers_[id].active) drop(*consumers_[id].active);
            consumers_[id].active.reset();
        }
        cycle_ = false;
        next_cycle_ = Clock::time_point{};
        glide_until_ = Clock::time_point{};
        return okay;
    }
    void loop() {
        for (;;) {
            std::unique_lock<std::mutex> lock(state_.mutex);
            auto now = Clock::now();
            bool expired = state_.skating && now >= state_.deadline;
            if (expired) {
                state_.skating = false;
                if (state_.tap) drop(*state_.tap);
                state_.tap.reset();
                if (state_.skill) drop(*state_.skill);
                state_.skill.reset();
                clear_plans(state_,output_);
                ++state_.generation;
            }
            if (state_.quit || observed_generation_ != state_.generation) {
                bool shutdown = state_.quit;
                observed_generation_ = state_.generation;
                lock.unlock();
                if (!release_all()) throw std::runtime_error("touch_release_failed");
                lock.lock();
                state_.idle = true;
                state_.cv.notify_all();
                if (expired) { output_.send("EXPIRED"); output_.send("STOPPED"); }
                if (shutdown || state_.quit) return;
            }
            // Standalone taps are allowed during the boss fight even when
            // the repeating ground gesture is stopped.
            now = Clock::now();
            bool advanced = false;
            for (int id : plan_fingers) {
                // Finger 5 is plan-only. Its due action is consumed in this iteration.
                std::optional<Shared::Tap> board_tap;
                auto& pending = id == 3 ? state_.tap : id == 4 ? state_.skill : board_tap;
                auto& consumer = consumers_[id];
                auto& plan = state_.plans[id];
                if (consumer.active && consumer.active->index >= 0 &&
                    consumer.active->revision != plan.revision) {
                    lock.unlock();
                    touch_up(id);
                    drop(*consumer.active);
                    consumer.active.reset();
                    advanced = true;
                    break;
                }
                if (!pending && !consumer.active && plan.begin < plan.end) {
                    auto& action = plan.actions[plan.begin];
                    if (timestamp_us() >= action.deadline_us) {
                        drop(action);
                        ++plan.begin;
                        advanced = true;
                        break;
                    }
                    if (timestamp_us() >= action.at_us) pending = std::move(plan.actions[plan.begin++]);
                }
                if (pending && (now >= pending->deadline ||
                    (pending->deadline_us && timestamp_us() >= pending->deadline_us))) {
                    if (pending->glide) {
                        state_.skating = false;
                        ++state_.generation;
                    }
                    drop(*pending);
                    pending.reset();
                }
                if (consumer.active && now >= consumer.release_at) {
                    lock.unlock();
                    touch_up(id);
                    const auto& tap = *consumer.active;
                    if (tap.index >= 0) output_.send("PLAN_RELEASED " + std::to_string(id) + " " +
                        std::to_string(tap.revision) + " " + std::to_string(tap.index) + " " + std::to_string(timestamp_us()));
                    else output_.send((tap.id.empty() ? "TAPPED " : "TAPPED2 " + tap.id + " ") +
                                      std::to_string(tap.x) + " " + std::to_string(tap.y));
                    consumer.active.reset();
                    advanced = true;
                    break;
                }
                if (pending && !consumer.active) {
                    consumer.active = *pending;
                    pending.reset();
                    state_.idle = false;
                    lock.unlock();
                    const auto& tap = *consumer.active;
                    if (tap.glide) {
                        if (held_[2]) touch_up(2);
                        if (jump_two_) touch_up(1);
                        if ((!jump_two_ || tap.x < 300) && !held_[1]) {
                            touch_down(1,190,620);
                            touch_down(1,190,700);
                        }
                        cycle_ = false;
                        next_cycle_ = Clock::time_point{};
                        glide_until_ = Clock::now()+350ms;
                    }
                    auto [start,end] = touch_down(id,tap.x,tap.y);
                    if (tap.index >= 0) output_.send("PLAN_PRESSED " + std::to_string(id) + " " +
                        std::to_string(tap.revision) + " " + std::to_string(tap.index) + " " +
                        std::to_string(start) + " " + std::to_string(end));
                    else if (!tap.id.empty()) output_.send("PRESSED " + tap.id + " " + std::to_string(start) + " " + std::to_string(end));
                    consumer.release_at = Clock::now()+std::chrono::milliseconds(tap.hold_ms);
                    advanced = true;
                    break;
                }
            }
            if (advanced) continue;
            if (!state_.skating && !state_.tap && !state_.skill &&
                !held_[3] && !held_[4] && !held_[5] && !has_plans()) {
                state_.idle = true;
                state_.cv.notify_all();
                state_.cv.wait(lock, [this] { return state_.quit || state_.skating || state_.tap.has_value() || state_.skill.has_value() || has_plans() || state_.generation != observed_generation_; });
                continue;
            }
            if (!state_.skating) {
                state_.cv.wait_until(lock,next_release(Clock::time_point::max()));
                continue;
            }
            if (now < glide_until_) {
                auto wake = std::min(glide_until_,state_.deadline);
                state_.cv.wait_until(lock,next_release(wake));
                continue;
            }
            state_.idle = false;
            const auto cycle_ms = jump_two_ ? 480ms : slide_ ? 300ms : 104ms;
            const auto* steps = jump_two_ ? jump_steps_ : steps_;
            const unsigned count = jump_two_ ? 6 : 7;
            if (!cycle_) {
                cycle_ = true; step_ = 0;
                cycle_base_ = next_cycle_ == Clock::time_point{} || now-next_cycle_ > cycle_ms ? now : next_cycle_;
                cycle_start_us_ = timestamp_us();
            }
            const auto due = cycle_base_ + (step_ < count ? std::chrono::milliseconds(steps[step_].ms) : cycle_ms);
            if (now >= due) {
                bool announce = state_.announce_start && step_ == 0;
                std::string start_id = announce ? state_.start_id : "";
                if (announce) state_.announce_start = false;
                unsigned step = step_++;
                lock.unlock();
                if (step < count) {
                    const auto& e = steps[step];
                    if (e.down) {
                        if (jump_two_ && e.id == 2) touch_up(1);
                        auto [down_start, down_end] = touch_down(e.id,e.x,e.y);
                        if (announce) output_.send(start_id.empty() ? "STARTED" : "STARTED " + start_id + " " + std::to_string(down_start) + " " + std::to_string(down_end));
                    } else touch_up(e.id);
                } else {
                    touch_up(1);
                    output_.send("CYCLE " + std::to_string(++cycle_count_) + " " +
                                 std::to_string(cycle_start_us_) + " " + std::to_string(timestamp_us()), true);
                    cycle_ = false;
                    next_cycle_ = cycle_base_+cycle_ms;
                }
                continue;
            }
            auto wake = std::min(due, state_.deadline);
            state_.cv.wait_until(lock,next_release(wake));
        }
    }
};

int wmain(int argc, wchar_t** argv) {
    std::wstring root;
    int index = -1;
    bool mock = false;
    bool slide = false;
    bool jump_two = false;
    bool swipe = false;
    int swipe_points[4]{};
    for (int i = 1; i < argc; ++i) {
        std::wstring option = argv[i];
        if (option == L"--folder" && i+1 < argc) root = argv[++i];
        else if (option == L"--index" && i+1 < argc) { try { index = std::stoi(argv[++i]); } catch (...) { return 2; } }
        else if (option == L"--mock") mock = true;
        else if (option == L"--slide") slide = true;
        else if (option == L"--jump-two-slide") jump_two = true;
        else if (option == L"--swipe" && i+4 < argc) {
            swipe = true;
            for (int j=0;j<4;++j) {
                try { swipe_points[j] = std::stoi(argv[++i]); } catch (...) { return 2; }
                if (swipe_points[j]<0 || swipe_points[j]>=(j%2 ? 720 : 1280)) return 2;
            }
        }
        else return 2;
    }
    if (!mock && (root.empty() || index < 0)) return 2;
    // Intentional process-lifetime ownership: a blocked stdout reader cannot
    // block touch cancellation or the process from exiting.
    auto output = new Output();
    try {
        // MuMu's DLL may write human-readable diagnostic output.  The protocol
        // owns the duplicated handle above; the DLL's normal stdout goes to
        // stderr so SDK messages can never corrupt protocol framing.
        fflush(stdout);
        if (_dup2(_fileno(stderr), _fileno(stdout)) != 0 ||
            !SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE)))
            throw std::runtime_error("cannot redirect SDK diagnostics");
        MuMu device(root,index,mock);
        if (swipe) {
            int error = 0;
            for (int step=0;step<=24 && !error;++step) {
                error = device.down(3,swipe_points[0]+(swipe_points[2]-swipe_points[0])*step/24,
                                      swipe_points[1]+(swipe_points[3]-swipe_points[1])*step/24);
                std::this_thread::sleep_for(20ms);
            }
            const int released = device.up(3);
            output->send(error || released ? "ERROR swipe_failed" : "SWIPED");
            output->drain_for(1s);
            return error || released ? 1 : 0;
        }
        Shared state;
        Engine engine(device,state,*output,slide,jump_two);
        std::thread worker([&] { engine.run(); });
        output->send("READY");
        output->send("CAPS PLAN1");
        output->send("CAPS PLAN5");
        std::string line;
        while (std::getline(std::cin,line)) {
            if (line.size() > 2048) { output->send("ERROR line_too_long"); continue; }
            std::istringstream input(line);
            std::string command, extra;
            input >> command;
            if (command == "START" || command == "RENEW") {
                int lease = 0;
                std::string seq;
                if (!(input >> lease) || lease < 50 || lease > 3000) { output->send("ERROR invalid_lease"); continue; }
                if (input >> seq) {
                    if (command != "START" || (input >> extra) || seq.size() > 20 ||
                        !std::all_of(seq.begin(), seq.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
                        output->send("ERROR invalid_start_id"); continue;
                    }
                }
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.fatal || state.quit) { output->send("ERROR unavailable"); continue; }
                if (command == "RENEW" && !state.skating) { output->send("ERROR not_started"); continue; }
                bool already = state.skating;
                state.skating = true;
                state.idle = false;
                state.deadline = Clock::now() + std::chrono::milliseconds(lease);
                if (command == "START") {
                    if (already && seq.empty()) output->send("STARTED");
                    else { state.announce_start = true; state.start_id = seq; }
                }
                state.cv.notify_all();
            } else if (command == "STOP") {
                std::string seq, trailing;
                int64_t barrier = 0;
                if (input >> seq) {
                    if (seq.empty() || seq.size() > 20 ||
                        !std::all_of(seq.begin(), seq.end(), [](unsigned char c) { return c >= '0' && c <= '9'; })) {
                        output->send("ERROR invalid_stop");
                        continue;
                    }
                    if (!(input >> std::ws).eof() && (!(input >> barrier) || barrier < 0 || (input >> trailing))) {
                        output->send("ERROR invalid_stop"); continue;
                    }
                }
                std::unique_lock<std::mutex> lock(state.mutex);
                state.skating = false;
                if (state.tap) output->send(tap_dropped(*state.tap));
                state.tap.reset();
                if (state.skill) output->send(tap_dropped(*state.skill));
                state.skill.reset();
                clear_plans(state,*output,barrier);
                state.announce_start = false;
                ++state.generation;
                state.cv.notify_all();
                state.cv.wait(lock,[&] { return state.idle || state.fatal; });
                if (state.idle) output->send(seq.empty() ? "STOPPED" : "STOPPED " + seq);
                else output->send("ERROR touch_release_failed");
            } else if (command == "PLAN" || command == "CANCEL") {
                int finger = 0, count = 0;
                int64_t revision = 0, captured = 0, expires = 0;
                if (!(input >> finger >> revision) || finger < plan_fingers.front() ||
                    finger > plan_fingers.back() || revision <= 0) {
                    output->send("ERROR invalid_plan"); continue;
                }
                Shared::Plan candidate;
                candidate.revision = revision;
                bool valid = true;
                if (command == "PLAN") {
                    const auto horizon_us = finger == 5 ? board_plan_horizon_us : plan_admission_us;
                    valid = bool(input >> captured >> expires >> count) && captured > 0 &&
                        expires > captured && expires-captured <= horizon_us && count > 0 &&
                        static_cast<size_t>(count) <= candidate.actions.size();
                    int64_t previous_up_offset = 0;
                    for (int i=0;valid && i<count;++i) {
                        auto& action = candidate.actions[i];
                        valid = bool(input >> action.at_us >> action.x >> action.y >> action.hold_ms >> action.deadline_us) &&
                            action.at_us >= captured && action.at_us < expires &&
                            action.at_us-captured >= previous_up_offset &&
                            action.deadline_us > action.at_us && action.deadline_us <= expires &&
                            action.x >= 0 && action.x < 1280 && action.y >= 0 && action.y < 720 &&
                            action.hold_ms >= 1 && action.hold_ms <= 200;
                        action.revision = revision; action.index = i; action.finger = finger;
                        if (valid) previous_up_offset = action.at_us-captured+action.hold_ms*1000;
                    }
                    candidate.end = valid ? static_cast<size_t>(count) : 0;
                }
                if (!valid || (input >> extra)) { output->send("ERROR invalid_plan"); continue; }
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.fatal || state.quit) { output->send("ERROR unavailable"); continue; }
                auto& plan = state.plans[finger];
                const auto now_us = timestamp_us();
                if (revision <= state.plan_floor || revision <= plan.revision ||
                    (command == "PLAN" && (captured > now_us || expires <= now_us ||
                                          now_us-captured >= plan_admission_us))) {
                    output->send("PLAN_REJECTED " + std::to_string(finger) + " " + std::to_string(revision)); continue;
                }
                const auto now = Clock::now();
                for (size_t i=0; i<candidate.end; ++i) {
                    auto& action = candidate.actions[i];
                    action.deadline = now+std::chrono::microseconds(std::max<int64_t>(0,action.deadline_us-now_us));
                }
                while (plan.begin < plan.end) output->send(tap_dropped(plan.actions[plan.begin++]));
                auto* pending = finger == 3 ? &state.tap : finger == 4 ? &state.skill : nullptr;
                if (pending && *pending) { output->send(tap_dropped(**pending)); pending->reset(); }
                plan = std::move(candidate);
                state.idle = false;
                output->send(std::string(command == "PLAN" ? "PLAN_ACCEPTED " : "PLAN_CLEARED ") +
                             std::to_string(finger) + " " + std::to_string(revision));
                state.cv.notify_all();
            } else if (command == "TAP") {
                int x = -1,y = -1;
                if (!(input >> x >> y) || (input >> extra) || x<0 || x>=1280 || y<0 || y>=720) {
                    output->send("ERROR invalid_tap"); continue;
                }
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.fatal || state.quit) { output->send("ERROR unavailable"); continue; }
                if (state.tap) output->send(tap_dropped(*state.tap));
                state.tap = Shared::Tap{x,y,Clock::now()+200ms};
                state.idle = false;
                state.cv.notify_all();
            } else if (command == "TAP2") {
                std::string seq, trailing;
                int x = -1, y = -1, hold = 0, priority = 0;
                int64_t deadline_us = 0;
                if (!(input >> seq >> x >> y >> hold >> deadline_us >> priority) || (input >> trailing) ||
                    seq.empty() || seq.size() > 20 ||
                    !std::all_of(seq.begin(), seq.end(), [](unsigned char c) { return c >= '0' && c <= '9'; }) ||
                    x<0 || x>=1280 || y<0 || y>=720 || hold<1 || hold>200 || deadline_us<=0 ||
                    (priority < 0 || priority > 2)) {
                    output->send("ERROR invalid_tap2"); continue;
                }
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.fatal || state.quit) { output->send("ERROR unavailable"); continue; }
                auto& pending = priority == 0 ? state.skill : state.tap;
                if (pending) output->send(tap_dropped(*pending));
                if (priority == 1) {
                    clear_plans(state,*output);
                    if (state.skill) output->send(tap_dropped(*state.skill));
                    state.skill.reset();
                    state.skating = false;
                    state.announce_start = false;
                    ++state.generation;
                }
                if (priority == 2) {
                    state.skating = true;
                    state.deadline = Clock::now()+500ms;
                }
                pending = Shared::Tap{x,y,Clock::now()+200ms,seq,hold,deadline_us,priority==2};
                state.idle = false;
                state.cv.notify_all();
            } else if (command == "QUIT" && !(input >> extra)) {
                break;
            } else {
                output->send("ERROR invalid_command");
            }
        }
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.quit = true;
            state.skating = false;
            if (state.tap) output->send(tap_dropped(*state.tap));
            state.tap.reset();
            if (state.skill) output->send(tap_dropped(*state.skill));
            state.skill.reset();
            clear_plans(state,*output);
            state.cv.notify_all();
        }
        worker.join();
        if (!state.fatal) output->send("STOPPED");
        output->send("BYE");
        output->drain_for(500ms);
        return state.fatal ? 1 : 0;
    } catch (const std::exception& e) {
        output->send(std::string("ERROR ") + e.what());
        output->drain_for(500ms);
        return 1;
    }
}

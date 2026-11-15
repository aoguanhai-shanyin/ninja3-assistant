using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;

namespace Ninja3Assistant;

public sealed partial class MainWindow
{
    private Process? _videoWorker;
    private readonly DispatcherTimer _videoPlayTimer = new();
    private readonly DispatcherTimer _videoSeekTimer = new() { Interval = TimeSpan.FromMilliseconds(150) };
    private readonly SemaphoreSlim _videoWriteLock = new(1, 1);
    private bool _openingVideo, _videoPlaying, _videoAwaitingFrame, _updatingVideoTimeline;
    private bool _videoSeekPending;
    private int _videoFrame, _videoTotalFrames, _pendingVideoFrame;
    private long _videoRequestId;
    private double _videoFps, _videoDuration;
    private bool VideoDebugBusy => _videoWorker is not null || _openingVideo;

    private void InitializeVideoDebug()
    {
        _videoPlayTimer.Tick += async (_, _) =>
        {
            if (!_videoPlaying || _videoAwaitingFrame || _videoSeekPending) return;
            if (_videoFrame >= _videoTotalFrames - 1) { PauseVideo(); return; }
            await RequestVideoFrameAsync(_videoFrame + 1);
        };
        _videoSeekTimer.Tick += async (_, _) =>
        {
            _videoSeekTimer.Stop();
            _videoSeekPending = false;
            await RequestVideoFrameAsync(_pendingVideoFrame, supersede: true);
        };
    }

    private void VideoNav_Click(object sender, RoutedEventArgs e)
    {
        TaskPage.Visibility = SettingsPage.Visibility = Visibility.Collapsed;
        VideoPage.Visibility = Visibility.Visible;
        VideoNav.Background = new SolidColorBrush(Windows.UI.Color.FromArgb(255, 38, 52, 81));
        TaskNav.Background = SettingsNav.Background = new SolidColorBrush(Windows.UI.Color.FromArgb(0, 0, 0, 0));
    }

    private void UpdateVideoDebugAvailability(bool otherWorkerBusy)
    {
        VideoImportButton.IsEnabled = !otherWorkerBusy && !_openingVideo;
        VideoCloseButton.IsEnabled = _videoWorker is not null;
        bool ready = _videoWorker is not null && _videoTotalFrames > 0 && !_openingVideo;
        VideoTimeline.IsEnabled = ready;
        VideoPlayButton.IsEnabled = ready && (_videoPlaying || !_videoAwaitingFrame);
        VideoPreviousButton.IsEnabled = ready && !_videoAwaitingFrame && _videoFrame > 0;
        VideoNextButton.IsEnabled = ready && !_videoAwaitingFrame && _videoFrame < _videoTotalFrames - 1;
        VideoPlayButton.Content = _videoPlaying ? "暂停" : "播放";
    }

    private async void VideoImport_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is not null || _scanning || _openingVideo) return;
        PauseVideo();
        var picker = new FileOpenPicker();
        foreach (string extension in new[] { ".mp4", ".m4s", ".mkv", ".avi", ".mov", ".webm" })
            picker.FileTypeFilter.Add(extension);
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
        try
        {
            var file = await picker.PickSingleFileAsync();
            if (file is not null) await LoadVideoDebugAsync(file.Path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Runtime.InteropServices.COMException)
        {
            VideoStatusText.Text = "无法选择视频：" + error.Message;
        }
    }

    private async Task LoadVideoDebugAsync(string path)
    {
        if (_worker is not null || _scanning || _openingVideo) return;
        CloseVideoDebug();
        VideoNav_Click(this, new RoutedEventArgs());
        if (!File.Exists(path)) { VideoStatusText.Text = "视频文件不存在：" + path; return; }
        string executable = Path.Combine(AppContext.BaseDirectory, "Ninja3Worker.exe");
        if (!File.Exists(executable)) { VideoStatusText.Text = "缺少运行组件 Ninja3Worker.exe。"; return; }
        _openingVideo = true;
        Title = "忍3助手 · 离线调试 · " + Path.GetFileName(path);
        _videoTotalFrames = 0;
        _videoFrame = 0;
        VideoFrameImage.Source = null;
        VideoPlaceholder.Visibility = Visibility.Visible;
        VideoFileText.Text = path;
        VideoBoardText.Text = "板子：等待识别";
        VideoStepText.Text = "流程：--";
        VideoActionText.Text = "建议动作：--";
        VideoStatusText.Text = "正在读取视频并初始化识别…";
        SetBusy(_worker is not null);
        try
        {
            var info = new ProcessStartInfo(executable)
            {
                UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
                WorkingDirectory = AppContext.BaseDirectory
            };
            info.ArgumentList.Add("--mode"); info.ArgumentList.Add("hayabusa-video");
            info.ArgumentList.Add("--video"); info.ArgumentList.Add(path);
            info.Environment["PYTHONIOENCODING"] = "utf-8";
            var process = Process.Start(info) ?? throw new InvalidOperationException("无法启动视频分析组件。");
            _videoWorker = process;
            SetBusy(_worker is not null);
            _ = ObserveVideoWorkerAsync(process);
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            _openingVideo = false;
            VideoStatusText.Text = "视频打开失败：" + error.Message;
            SetBusy(_worker is not null);
        }
        await Task.CompletedTask;
    }

    private async Task ObserveVideoWorkerAsync(Process process)
    {
        async Task ReadAsync(StreamReader reader, bool isError)
        {
            string? line;
            while ((line = await reader.ReadLineAsync()) is not null)
            {
                string captured = line;
                if (isError)
                {
                    DispatcherQueue.TryEnqueue(() =>
                    {
                        if (ReferenceEquals(_videoWorker, process)) AddLog("视频分析：" + captured);
                    });
                    continue;
                }
                if (!captured.TrimStart().StartsWith('{')) continue;
                try
                {
                    using var document = JsonDocument.Parse(captured);
                    var message = document.RootElement.Clone();
                    DispatcherQueue.TryEnqueue(async () =>
                    {
                        if (!ReferenceEquals(_videoWorker, process)) return;
                        try { await HandleVideoMessageAsync(process, message); }
                        catch (Exception error) when (error is JsonException or KeyNotFoundException or
                            InvalidOperationException or FormatException or System.Runtime.InteropServices.COMException)
                        {
                            if (!ReferenceEquals(_videoWorker, process)) return;
                            if (message.GetProperty("event").GetString() == "video_frame" &&
                                message.GetProperty("value").TryGetProperty("request_id", out var failedRequest) &&
                                failedRequest.GetInt64() != _videoRequestId) return;
                            _videoAwaitingFrame = false;
                            PauseVideo();
                            VideoStatusText.Text = "无法读取分析画面：" + error.Message;
                        }
                    });
                }
                catch (JsonException error)
                {
                    DispatcherQueue.TryEnqueue(() =>
                    {
                        if (ReferenceEquals(_videoWorker, process)) VideoStatusText.Text = "视频分析返回了无效数据：" + error.Message;
                    });
                }
            }
        }
        try
        {
            await Task.WhenAll(ReadAsync(process.StandardOutput, false), ReadAsync(process.StandardError, true), process.WaitForExitAsync());
            int code = process.ExitCode;
            DispatcherQueue.TryEnqueue(() =>
            {
                if (!ReferenceEquals(_videoWorker, process)) return;
                _videoWorker = null;
                _openingVideo = _videoAwaitingFrame = false;
                PauseVideo();
                if (code != 0) VideoStatusText.Text = "视频分析组件已退出，代码：" + code;
                else if (!VideoStatusText.Text.Contains("失败")) VideoStatusText.Text += " · 分析组件已结束，可重新导入。";
                process.Dispose();
                SetBusy(_worker is not null);
            });
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or ObjectDisposedException)
        {
            DispatcherQueue.TryEnqueue(() =>
            {
                if (!ReferenceEquals(_videoWorker, process)) return;
                VideoStatusText.Text = "视频分析已中断：" + error.Message;
                CloseVideoDebug();
            });
        }
    }

    private async Task HandleVideoMessageAsync(Process process, JsonElement message)
    {
        string? kind = message.GetProperty("event").GetString();
        var value = message.GetProperty("value");
        if (kind == "video_info")
        {
            _videoFps = value.GetProperty("fps").GetDouble();
            _videoDuration = value.GetProperty("duration").GetDouble();
            _videoTotalFrames = value.GetProperty("total_frames").GetInt32();
            _openingVideo = false;
            _videoPlayTimer.Interval = TimeSpan.FromSeconds(1 / Math.Clamp(_videoFps, 1, 120));
            _updatingVideoTimeline = true;
            VideoTimeline.Maximum = Math.Max(1, _videoTotalFrames - 1);
            VideoTimeline.Value = 0;
            _updatingVideoTimeline = false;
            VideoTimeText.Text = $"00:00.000 / {VideoTime(_videoDuration)} · 帧 1/{_videoTotalFrames}";
            VideoFileText.Text = $"{value.GetProperty("path").GetString()} · {_videoFps:F2} fps · {value.GetProperty("backend").GetString()}";
            SetBusy(_worker is not null);
            await RequestVideoFrameAsync(0);
        }
        else if (kind == "video_frame")
        {
            long request = value.GetProperty("request_id").GetInt64();
            if (request != _videoRequestId || _videoSeekPending) return;
            byte[] bytes = Convert.FromBase64String(value.GetProperty("jpeg_base64").GetString() ?? "");
            using var stream = new InMemoryRandomAccessStream();
            using (var writer = new DataWriter(stream.GetOutputStreamAt(0)))
            {
                writer.WriteBytes(bytes);
                await writer.StoreAsync();
                await writer.FlushAsync();
            }
            stream.Seek(0);
            var bitmap = new BitmapImage();
            await bitmap.SetSourceAsync(stream);
            if (!ReferenceEquals(_videoWorker, process) || request != _videoRequestId || _videoSeekPending) return;
            VideoFrameImage.Source = bitmap;
            VideoPlaceholder.Visibility = Visibility.Collapsed;
            _videoFrame = value.GetProperty("frame").GetInt32();
            _videoAwaitingFrame = false;
            _updatingVideoTimeline = true;
            VideoTimeline.Value = _videoFrame;
            _updatingVideoTimeline = false;
            VideoTimeText.Text = $"{VideoTime(value.GetProperty("seconds").GetDouble())} / {VideoTime(_videoDuration)} · 帧 {_videoFrame + 1}/{_videoTotalFrames}";
            var screen = value.GetProperty("screen");
            string board = VideoString(screen, "board_name");
            string phase = VideoString(screen, "phase_hint");
            bool active = screen.TryGetProperty("active", out var flag) && flag.ValueKind == JsonValueKind.True;
            double confidence = screen.TryGetProperty("board_confidence", out var score) ? score.GetDouble() : 0;
            VideoBoardText.Text = string.IsNullOrEmpty(board) ? active ? "板子：待判别 · 实时轨迹避让" : "板子：尚未确认隼白战斗" : $"{phase} · {board} · {confidence:P0}";
            var recipe = value.GetProperty("recipe");
            if (recipe.ValueKind == JsonValueKind.Object && recipe.TryGetProperty("mode_active", out var mode) && mode.ValueKind == JsonValueKind.True)
            {
                bool late = recipe.TryGetProperty("late_entry", out var entry) && entry.ValueKind == JsonValueKind.True;
                VideoStepText.Text = $"流程：第 {recipe.GetProperty("stage").GetInt32() + 1} 步 · 波次 {recipe.GetProperty("wave").GetInt32()} · {VideoString(recipe, "mode_state")}" + (late ? " · 中途进入，按当前波次避让" : "");
            }
            else VideoStepText.Text = "流程：等待板子确认或重新同步";
            var action = value.GetProperty("action");
            VideoActionText.Text = "建议动作：" + (action.ValueKind == JsonValueKind.Object ? VideoActionName(VideoString(action, "kind")) : "等待") + "（不发送操作）";
            bool resynced = value.TryGetProperty("resynced", out var synced) && synced.ValueKind == JsonValueKind.True;
            string reason = recipe.ValueKind == JsonValueKind.Object ? VideoString(recipe, "reason") : "";
            VideoStatusText.Text = $"{(resynced ? "已跳转并重新同步 · " : "")}分析 {value.GetProperty("analysis_ms").GetDouble():F1} ms · {reason}";
            if (_videoFrame >= _videoTotalFrames - 1) PauseVideo();
            UpdateVideoDebugAvailability(_worker is not null || _scanning);
        }
        else if (kind == "video_error")
        {
            if (value.TryGetProperty("request_id", out var id) && id.ValueKind == JsonValueKind.Number && id.GetInt64() != _videoRequestId) return;
            _videoAwaitingFrame = _openingVideo = false;
            PauseVideo();
            VideoStatusText.Text = "视频分析失败：" + VideoString(value, "message");
            SetBusy(_worker is not null);
        }
    }

    private async Task RequestVideoFrameAsync(int frame, bool supersede = false)
    {
        var process = _videoWorker;
        if (process is null || _videoTotalFrames <= 0 || (_videoAwaitingFrame && !supersede)) return;
        frame = Math.Clamp(frame, 0, _videoTotalFrames - 1);
        long request = ++_videoRequestId;
        _videoAwaitingFrame = true;
        VideoStatusText.Text = $"正在分析第 {frame + 1} 帧…";
        UpdateVideoDebugAvailability(_worker is not null || _scanning);
        await _videoWriteLock.WaitAsync();
        try
        {
            if (!ReferenceEquals(_videoWorker, process) || request != _videoRequestId) return;
            await process.StandardInput.WriteLineAsync(JsonSerializer.Serialize(new { command = "frame", frame, request_id = request }));
            await process.StandardInput.FlushAsync();
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or ObjectDisposedException)
        {
            if (!ReferenceEquals(_videoWorker, process)) return;
            _videoAwaitingFrame = false;
            PauseVideo();
            VideoStatusText.Text = "无法请求视频帧：" + error.Message;
        }
        finally { _videoWriteLock.Release(); }
    }

    private void VideoTimeline_ValueChanged(object sender, RangeBaseValueChangedEventArgs e)
    {
        if (_updatingVideoTimeline || _videoTotalFrames <= 0 || _videoWorker is null) return;
        PauseVideo();
        _videoRequestId++;
        _pendingVideoFrame = Math.Clamp((int)Math.Round(e.NewValue), 0, _videoTotalFrames - 1);
        _videoSeekPending = true;
        VideoStatusText.Text = "松开时间轴后分析当前帧并重新同步…";
        _videoSeekTimer.Stop();
        _videoSeekTimer.Start();
    }

    private async void VideoPrevious_Click(object sender, RoutedEventArgs e)
    {
        PauseVideo();
        await RequestVideoFrameAsync(_videoFrame - 1);
    }

    private async void VideoNext_Click(object sender, RoutedEventArgs e)
    {
        PauseVideo();
        await RequestVideoFrameAsync(_videoFrame + 1);
    }

    private async void VideoPlay_Click(object sender, RoutedEventArgs e)
    {
        if (_videoPlaying) { PauseVideo(); return; }
        if (_videoWorker is null || _videoTotalFrames <= 0) return;
        _videoPlaying = true;
        if (_videoFrame >= _videoTotalFrames - 1) await RequestVideoFrameAsync(0);
        _videoPlayTimer.Start();
        UpdateVideoDebugAvailability(_worker is not null || _scanning);
    }

    private void PauseVideo()
    {
        _videoPlaying = false;
        _videoPlayTimer.Stop();
        UpdateVideoDebugAvailability(_worker is not null || _scanning);
    }

    private void VideoClose_Click(object sender, RoutedEventArgs e)
    {
        CloseVideoDebug();
        VideoStatusText.Text = "视频已关闭；可导入另一段录像或启动正常助手。";
    }

    private void CloseVideoDebug(bool updateControls = true)
    {
        _videoRequestId++;
        _videoSeekTimer.Stop();
        _videoSeekPending = _videoPlaying = false;
        _videoPlayTimer.Stop();
        var process = _videoWorker;
        _videoWorker = null;
        _openingVideo = _videoAwaitingFrame = false;
        _videoTotalFrames = 0;
        if (process is not null)
        {
            try
            {
                if (!process.HasExited)
                {
                    process.StandardInput.WriteLine("{\"command\":\"stop\"}");
                    process.StandardInput.Flush();
                    process.StandardInput.Close();
                    if (!process.WaitForExit(10000)) process.Kill(true);
                }
            }
            catch (Exception error) when (error is IOException or InvalidOperationException or System.ComponentModel.Win32Exception) { }
            process.Dispose();
        }
        if (updateControls)
        {
            Title = "忍3助手";
            SetBusy(_worker is not null);
        }
    }

    private static string VideoString(JsonElement value, string name) =>
        value.TryGetProperty(name, out var property) && property.ValueKind == JsonValueKind.String ? property.GetString() ?? "" : "";

    private static string VideoTime(double seconds) => TimeSpan.FromSeconds(Math.Max(0, seconds)).ToString(@"mm\:ss\.fff");

    private static string VideoActionName(string kind) => kind switch
    {
        "jump" => "上跳", "up" => "翻上绳索", "down" => "向下翻绳", "drop" => "主动下降",
        "roll" => "绳下翻滚", "weapon" => "武器技能", _ => kind
    };
}

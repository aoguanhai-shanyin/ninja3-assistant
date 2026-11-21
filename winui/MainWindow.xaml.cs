using System.ComponentModel;
using System.Diagnostics;
using System.Text.Json.Serialization;
using System.Text;
using System.Text.Json;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Windows.Graphics;
using Windows.Storage.Pickers;

namespace Ninja3Assistant;

public sealed partial class MainWindow : Window
{
    private static readonly string[] BountyGrades = ["D", "C", "B", "A"];
    private static readonly string[] NewDailyTasks = ["pet_feed", "arena_reward", "landlady_quiz"];
    private const int DailyTasksVersion = 1;
    private readonly string _settingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Ninja3Assistant", "settings.json");
    private readonly Queue<string> _log = new();
    private readonly Dictionary<string, WorkerSession> _sessions = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Settings> _profiles = new(StringComparer.OrdinalIgnoreCase);
    private Settings _defaults = new();
    private InstanceOption? _selectedInstance;
    private WorkerSession? SelectedSession => _selectedInstance?.Session;
    private bool AnyWorker => _sessions.Values.Any(session => session.Process is not null);
    private bool _scanning;
    private bool _updatingInstances;
    private bool? _compactLayout;
    private int _restoredCount;
    private int? _savedIndex;
    private string? _savedName;
    private string? _savedFolder;
    private string? _scannedFolder;

    private sealed record Settings
    {
        public string Folder { get; set; } = @"D:\Program Files\Netease\MuMu";
        public int? Index { get; set; }
        public string? InstanceName { get; set; }
        public int JoinIntervalMs { get; set; } = 30;
        public bool ActivityEnabled { get; set; }
        public string ActivityMode { get; set; } = "daily";
        public bool WeekendEnabled { get; set; }
        public string WeekendMode { get; set; } = "daily";
        public bool XumiEnabled { get; set; }
        public bool AbyssEnabled { get; set; }
        public bool HideGameWindow { get; set; }
        public bool DailyEnabled { get; set; }
        public string[] DailyTasks { get; set; } = ["monthly", "limited", "blessing", "weapon", "curse", "bounty_shop", "daily_shop", "village", "house", "bounty", "bounty_s", "rank", .. NewDailyTasks];
        public int DailyTaskOptionsVersion { get; set; }
        public string BountyGrade { get; set; } = "D";
        public bool BountyActiveInvite { get; set; } = true;
        public bool BountyPassiveInvite { get; set; } = true;
        public string PetFeedReward { get; set; } = "exp";
        public bool CloseToTray { get; set; } = true;
        public bool RememberCloseAction { get; set; }
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public Dictionary<string, Settings>? InstanceProfiles { get; set; }
    }

    private sealed class WorkerSession
    {
        public Process? Process { get; set; }
        public string Mode { get; set; } = "";
        public bool Paused { get; set; }
        public bool Stopping { get; set; }
        public bool CommandPending { get; set; }
        public int Completed { get; set; }
        public string State { get; set; } = "未运行";
        public string Status { get; set; } = "可以启动当前实例。";
        public string Device { get; set; } = "识别设备：尚未检测";
        public string Probe { get; set; } = "";
        public string Rate { get; set; } = "--";
        public string Clicks { get; set; } = "0";
        public string Interval { get; set; } = "最近连点：--";
        public Queue<string> Log { get; } = new();
    }

    private sealed class InstanceOption : INotifyPropertyChanged
    {
        public required string Folder { get; init; }
        public required string Key { get; init; }
        public required int Index { get; init; }
        public required string Name { get; init; }
        public required bool Started { get; init; }
        public required WorkerSession Session { get; init; }
        public string DisplayName => $"{Name}  ·  #{Index}  ·  {(Session.Process is not null
            ? Session.State : Started ? "模拟器已启动" : "模拟器未启动")}";
        public event PropertyChangedEventHandler? PropertyChanged;
        public void Refresh() => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(DisplayName)));
    }

    public MainWindow()
    {
        InitializeComponent();
        ActivityTabs.SelectionChanged += ActivityTabs_SelectionChanged;
        ActivityTabs.SelectedItem = IntelligenceTab;
        WeekendTabs.SelectedItem = WeekendTab;
        FamilyTabs.SelectedItem = AbyssTab;
        InitializeVideoDebug();
        AppWindow.Resize(new SizeInt32(1080, 760));
        Closed += Window_Closed;
        LoadSettings();
        InitializeTrayIntegration();
        AddLog("准备就绪。");
        RestoreSessionCount();
        UpdateControls();
    }

    private void RestoreSessionCount()
    {
        string[] args = Environment.GetCommandLineArgs();
        int index = Array.IndexOf(args, "--resume-settlements");
        if (index < 0 || index + 1 >= args.Length || !int.TryParse(args[index + 1], out int count) || count < 0)
            return;
        _restoredCount = count;
        CountText.Text = count.ToString();
        AddLog("已保留更新前的结算计数：" + count + " 次。");
    }

    private void LoadSettings()
    {
        var settings = new Settings();
        try
        {
            if (File.Exists(_settingsPath))
                settings = JsonSerializer.Deserialize<Settings>(File.ReadAllText(_settingsPath)) ?? settings;
        }
        catch (Exception error) when (error is IOException or JsonException or UnauthorizedAccessException)
        {
            AddLog("设置读取失败，已使用默认值：" + error.Message);
        }
        _defaults = settings with { InstanceProfiles = null };
        if (settings.InstanceProfiles is not null)
            foreach (var (key, profile) in settings.InstanceProfiles)
                _profiles[key] = profile;
        FolderBox.Text = settings.Folder;
        LoadCloseBehaviorSettings(settings);
        _savedIndex = settings.Index;
        _savedName = settings.InstanceName;
        _savedFolder = settings.Folder;
        ApplyOptions(settings);
    }

    private void ApplyOptions(Settings settings)
    {
        JoinIntervalBox.Value = settings.JoinIntervalMs is >= 20 and <= 1000 ? settings.JoinIntervalMs : 30;
        ActivityEnabledSwitch.IsOn = settings.ActivityEnabled;
        ActivityModeBox.SelectedIndex = settings.ActivityMode == "continuous" ? 1 : 0;
        ActivityModeBox.IsEnabled = settings.ActivityEnabled;
        WeekendEnabledSwitch.IsOn = settings.WeekendEnabled;
        WeekendModeBox.SelectedIndex = settings.WeekendMode == "25" ? 1 : 0;
        WeekendModeBox.IsEnabled = settings.WeekendEnabled;
        XumiEnabledSwitch.IsOn = settings.XumiEnabled;
        AbyssEnabledSwitch.IsOn = settings.AbyssEnabled;
        HideGameWindowSwitch.IsOn = settings.HideGameWindow;
        DailyEnabledSwitch.IsOn = settings.DailyEnabled;
        var dailyTasks = settings.DailyTasks ?? [];
        if (settings.DailyTaskOptionsVersion < DailyTasksVersion)
            dailyTasks = dailyTasks.Union(NewDailyTasks).ToArray();
        foreach (var option in DailyOptionsPanel.Children.OfType<CheckBox>())
            option.IsChecked = dailyTasks.Contains((string)option.Tag);
        DailyOptionsControl.IsEnabled = settings.DailyEnabled;
        BountyGradeSlider.Value = Math.Max(0, Array.IndexOf(BountyGrades, settings.BountyGrade));
        BountyActiveInviteOption.IsChecked = settings.BountyActiveInvite;
        BountyPassiveInviteOption.IsChecked = settings.BountyPassiveInvite;
        UpdateBountyOptions();
        PetFeedRewardBox.SelectedItem = PetFeedRewardBox.Items.OfType<ComboBoxItem>()
            .FirstOrDefault(option => (string)option.Tag == settings.PetFeedReward) ?? PetFeedRewardBox.Items[0];
        PetFeedRewardBox.IsEnabled = PetFeedTaskOption.IsChecked == true;
    }

    private bool TrySettings(out Settings settings, bool requireRunning)
    {
        settings = new Settings { Folder = FolderBox.Text.Trim() };
        if (!Directory.Exists(settings.Folder))
        {
            ShowMessage("MuMu 目录不存在，请先在设备设置中选择正确目录。");
            return false;
        }
        if (InstanceBox.SelectedItem is not InstanceOption instance)
        {
            ScanText.Text = "请先选择一个 MuMu 实例。";
            ShowMessage(ScanText.Text);
            return false;
        }
        if (requireRunning && !instance.Started)
        {
            ScanText.Text = "所选模拟器尚未启动，请先在 MuMu 中启动并重新扫描。";
            ShowMessage(ScanText.Text);
            return false;
        }
        if (!ValidJoinInterval())
        {
            ShowMessage("抢队连点间隔请输入 20 到 1000 之间的整数毫秒。");
            return false;
        }
        settings = ReadOptions(instance);
        return true;
    }

    private bool ValidJoinInterval() => JoinIntervalBox.Value is >= 20 and <= 1000 &&
        JoinIntervalBox.Value == Math.Truncate(JoinIntervalBox.Value);

    private Settings ReadOptions(InstanceOption instance)
    {
        var settings = new Settings
        {
            Folder = instance.Folder,
            Index = instance.Index,
            InstanceName = instance.Name,
            JoinIntervalMs = ValidJoinInterval() ? (int)JoinIntervalBox.Value :
                _profiles.GetValueOrDefault(instance.Key, _defaults).JoinIntervalMs
        };
        settings.ActivityEnabled = ActivityEnabledSwitch.IsOn;
        settings.ActivityMode = ActivityModeBox.SelectedIndex == 1 ? "continuous" : "daily";
        settings.WeekendEnabled = WeekendEnabledSwitch.IsOn;
        settings.WeekendMode = WeekendModeBox.SelectedIndex == 1 ? "25" : "daily";
        settings.XumiEnabled = XumiEnabledSwitch.IsOn;
        settings.AbyssEnabled = AbyssEnabledSwitch.IsOn;
        settings.HideGameWindow = HideGameWindowSwitch.IsOn;
        settings.DailyEnabled = DailyEnabledSwitch.IsOn;
        settings.DailyTasks = DailyOptionsPanel.Children.OfType<CheckBox>()
            .Where(option => option.IsChecked == true).Select(option => (string)option.Tag).ToArray();
        settings.DailyTaskOptionsVersion = DailyTasksVersion;
        settings.BountyGrade = BountyGrades[(int)Math.Round(BountyGradeSlider.Value)];
        settings.BountyActiveInvite = BountyActiveInviteOption.IsChecked == true;
        settings.BountyPassiveInvite = BountyPassiveInviteOption.IsChecked == true;
        settings.PetFeedReward = (PetFeedRewardBox.SelectedItem as ComboBoxItem)?.Tag as string ?? "exp";
        settings.CloseToTray = _closeToTray;
        settings.RememberCloseAction = _rememberCloseAction;
        return settings;
    }

    private bool PersistSettings(Settings settings)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_settingsPath)!);
            string temp = _settingsPath + ".tmp";
            if (_selectedInstance is not null) _profiles[_selectedInstance.Key] = settings;
            File.WriteAllText(temp, JsonSerializer.Serialize(settings with { InstanceProfiles = _profiles },
                new JsonSerializerOptions { WriteIndented = true }));
            File.Move(temp, _settingsPath, true);
            _savedIndex = settings.Index;
            _savedName = settings.InstanceName;
            _savedFolder = settings.Folder;
            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            ShowMessage("保存设置失败：" + error.Message);
            return false;
        }
    }

    private async void Root_Loaded(object sender, RoutedEventArgs e)
    {
        if (sender is FrameworkElement root && root.XamlRoot is not null)
        {
            double scale = root.XamlRoot.RasterizationScale;
            var area = DisplayArea.GetFromWindowId(AppWindow.Id, DisplayAreaFallback.Primary).WorkArea;
            AppWindow.Resize(new SizeInt32(Math.Min((int)(1050 * scale), area.Width),
                                           Math.Min((int)(750 * scale), area.Height)));
        }
        string[] args = Environment.GetCommandLineArgs();
        int videoIndex = Array.IndexOf(args, "--debug-video");
        if (videoIndex >= 0 && videoIndex + 1 < args.Length)
            await LoadVideoDebugAsync(args[videoIndex + 1]);
        else
            await ScanInstancesAsync();
    }

    private void Root_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        bool compact = e.NewSize.Width < 720;
        if (_compactLayout == compact || SidebarColumn is null) return;
        _compactLayout = compact;
        SidebarColumn.Width = new GridLength(compact ? 62 : 190);
        Sidebar.Padding = compact ? new Thickness(10, 18, 10, 18) : new Thickness(20, 26, 20, 26);
        var visibility = compact ? Visibility.Collapsed : Visibility.Visible;
        BrandText.Visibility = NavHeading.Visibility = SidebarFooter.Visibility = visibility;
        TaskNavLabel.Visibility = SettingsNavLabel.Visibility = VideoNavLabel.Visibility = visibility;
        foreach (var button in new[] { TaskNav, SettingsNav, VideoNav })
        {
            button.Padding = new Thickness(compact ? 0 : 14, 10, compact ? 0 : 14, 10);
            button.HorizontalContentAlignment = compact ? HorizontalAlignment.Center : HorizontalAlignment.Left;
        }
        TaskToolbar.Padding = compact ? new Thickness(16, 12, 16, 12) : new Thickness(24, 14, 24, 14);
        TaskContent.Padding = SettingsContent.Padding = compact
            ? new Thickness(16, 18, 16, 18) : new Thickness(34, 30, 34, 30);
    }

    private void Instance_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_updatingInstances) SelectInstance(InstanceBox.SelectedItem as InstanceOption);
    }

    private void SelectInstance(InstanceOption? instance)
    {
        if (_selectedInstance?.Key != instance?.Key)
        {
            if (_selectedInstance is { } previous)
                _profiles[previous.Key] = ReadOptions(previous);
            _selectedInstance = instance;
            if (instance is not null)
                ApplyOptions(_profiles.GetValueOrDefault(instance.Key, _defaults));
        }
        else
        {
            _selectedInstance = instance;
        }
        if (instance is not null && _restoredCount > 0)
        {
            instance.Session.Completed = _restoredCount;
            _restoredCount = 0;
        }
        SettingsInstanceText.Text = instance is null ? "尚未选择实例" : $"{instance.Name}（#{instance.Index}）";
        RenderSelectedSession();
        UpdateControls();
    }

    private async void Scan_Click(object sender, RoutedEventArgs e) => await ScanInstancesAsync();

    private async Task ScanInstancesAsync()
    {
        if (_exiting || _scanning || VideoDebugBusy) return;
        string folder = FolderBox.Text.Trim();
        string manager = Path.Combine(folder, "nx_main", "MuMuManager.exe");
        var previous = InstanceBox.SelectedItem as InstanceOption;
        bool sameFolder = string.Equals(folder, _scannedFolder, StringComparison.OrdinalIgnoreCase);
        int? preferredIndex = sameFolder ? previous?.Index ?? _savedIndex :
            string.Equals(folder, _savedFolder, StringComparison.OrdinalIgnoreCase) ? _savedIndex : null;
        string? preferredName = sameFolder ? previous?.Name ?? _savedName :
            string.Equals(folder, _savedFolder, StringComparison.OrdinalIgnoreCase) ? _savedName : null;
        _scanning = true;
        UpdateControls();
        ScanText.Text = "正在扫描 MuMu 多开实例…";
        try
        {
            if (!File.Exists(manager)) throw new FileNotFoundException("未找到 MuMuManager.exe，请检查安装目录。", manager);
            var info = new ProcessStartInfo(manager)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
            info.ArgumentList.Add("info");
            info.ArgumentList.Add("--vmindex");
            info.ArgumentList.Add("all");
            using var process = Process.Start(info) ?? throw new InvalidOperationException("无法启动 MuMu 多开器查询。");
            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(8));
            try { await process.WaitForExitAsync(timeout.Token); }
            catch (OperationCanceledException)
            {
                if (!process.HasExited) process.Kill(true);
                throw new TimeoutException("扫描超时，请检查 MuMu 多开器。");
            }
            string output = await outputTask;
            string error = await errorTask;
            if (process.ExitCode != 0) throw new InvalidOperationException("MuMu 查询失败：" + error.Trim());
            using var json = JsonDocument.Parse(output.TrimStart('\uFEFF', ' ', '\r', '\n', '\t'));
            var instances = new List<InstanceOption>();
            foreach (var item in json.RootElement.EnumerateObject())
            {
                if (!int.TryParse(item.Name, out int index) || item.Value.ValueKind != JsonValueKind.Object) continue;
                var value = item.Value;
                string name = value.TryGetProperty("name", out var nameValue) && nameValue.ValueKind == JsonValueKind.String
                    ? nameValue.GetString() ?? "模拟器" : "模拟器";
                bool started = value.TryGetProperty("is_process_started", out var running) && running.ValueKind == JsonValueKind.True
                    && value.TryGetProperty("is_android_started", out var android) && android.ValueKind == JsonValueKind.True;
                string key = Path.TrimEndingDirectorySeparator(Path.GetFullPath(folder)) + "|" + index;
                if (!_sessions.TryGetValue(key, out var session))
                    _sessions[key] = session = new WorkerSession();
                instances.Add(new InstanceOption
                {
                    Folder = folder, Key = key, Index = index, Name = name, Started = started, Session = session
                });
            }
            instances.Sort((a, b) => a.Index.CompareTo(b.Index));
            _scannedFolder = folder;
            _updatingInstances = true;
            InstanceBox.ItemsSource = instances;
            InstanceBox.SelectedItem = instances.FirstOrDefault(item => item.Index == preferredIndex &&
                (sameFolder || preferredName is null || item.Name == preferredName)) ??
                instances.FirstOrDefault(item => item.Started) ?? instances.FirstOrDefault();
            _updatingInstances = false;
            SelectInstance(InstanceBox.SelectedItem as InstanceOption);
            int runningCount = instances.Count(item => item.Started);
            ScanText.Text = instances.Count == 0 ? "未发现模拟器实例，请先在 MuMu 多开器中创建。" :
                $"已发现 {instances.Count} 个实例，其中 {runningCount} 个运行中。" +
                (InstanceBox.SelectedItem is null ? "请选择要控制的实例。" : "");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException or
                                      InvalidOperationException or TimeoutException or System.ComponentModel.Win32Exception)
        {
            if (!sameFolder && !AnyWorker)
            {
                _scannedFolder = null;
                InstanceBox.ItemsSource = null;
            }
            ScanText.Text = "扫描失败：" + error.Message;
            AddLog(ScanText.Text);
        }
        finally
        {
            _scanning = false;
            UpdateControls();
        }
    }

    private async void Start_Click(object sender, RoutedEventArgs e) => await StartWorkerAsync("run");
    private async void DebugStart_Click(object sender, RoutedEventArgs e) => await StartWorkerAsync("hayabusa-debug");
    private async void Probe_Click(object sender, RoutedEventArgs e) => await StartWorkerAsync("probe");

    private async Task StartWorkerAsync(string mode)
    {
        if (_exiting || _scanning || VideoDebugBusy) return;
        if (!string.Equals(FolderBox.Text.Trim(), _scannedFolder, StringComparison.OrdinalIgnoreCase))
        {
            await ScanInstancesAsync();
            if (_exiting || !string.Equals(FolderBox.Text.Trim(), _scannedFolder, StringComparison.OrdinalIgnoreCase)) return;
        }
        if (SelectedSession is not { Process: null } session ||
            !TrySettings(out var settings, requireRunning: mode != "run"))
            return;
        if (mode == "run" && settings.DailyEnabled && settings.DailyTasks.Length == 0)
        {
            ShowMessage("请至少选择一项每日清理任务。");
            return;
        }
        if (!PersistSettings(settings)) return;
        string executable = Path.Combine(AppContext.BaseDirectory, "Ninja3Worker.exe");
        if (!File.Exists(executable))
        {
            ShowMessage("缺少运行组件 Ninja3Worker.exe，请检查程序文件是否完整。");
            return;
        }
        try
        {
            var info = new ProcessStartInfo(executable)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                WorkingDirectory = AppContext.BaseDirectory
            };
            info.ArgumentList.Add("--mode");
            info.ArgumentList.Add(mode);
            if (settings.HideGameWindow && mode != "probe") info.ArgumentList.Add("--hide-game-window");
            info.ArgumentList.Add("--folder");
            info.ArgumentList.Add(settings.Folder);
            info.ArgumentList.Add("--index");
            info.ArgumentList.Add(settings.Index!.Value.ToString());
            info.ArgumentList.Add("--join-interval-ms");
            info.ArgumentList.Add(settings.JoinIntervalMs.ToString());
            if (mode == "run" && settings.ActivityEnabled)
            {
                info.ArgumentList.Add("--activity-enabled");
                info.ArgumentList.Add("--activity-mode");
                info.ArgumentList.Add(settings.ActivityMode);
            }
            info.Environment["PYTHONIOENCODING"] = "utf-8";
            if (mode == "run" && settings.WeekendEnabled)
            {
                info.ArgumentList.Add("--weekend-enabled");
                info.ArgumentList.Add("--weekend-mode");
                info.ArgumentList.Add(settings.WeekendMode);
            }

            if (mode == "run" && settings.XumiEnabled)
                info.ArgumentList.Add("--xumi-enabled");
            if (mode == "run" && settings.AbyssEnabled)
                info.ArgumentList.Add("--abyss-enabled");
            if (mode == "run" && settings.DailyEnabled)
            {
                info.ArgumentList.Add("--daily-tasks");
                foreach (string task in settings.DailyTasks) info.ArgumentList.Add(task);
                bool regularBounty = settings.DailyTasks.Contains("bounty");
                if (regularBounty)
                {
                    info.ArgumentList.Add("--bounty-grade");
                    info.ArgumentList.Add(settings.BountyGrade);
                }
                if (regularBounty || settings.DailyTasks.Contains("bounty_s"))
                {
                    if (!settings.BountyActiveInvite) info.ArgumentList.Add("--no-bounty-active-invite");
                    if (!settings.BountyPassiveInvite) info.ArgumentList.Add("--no-bounty-passive-invite");
                }
                if (settings.DailyTasks.Contains("pet_feed"))
                {
                    info.ArgumentList.Add("--pet-feed-reward");
                    info.ArgumentList.Add(settings.PetFeedReward);
                }
            }
            var process = Process.Start(info) ?? throw new InvalidOperationException("无法启动运行组件");
            session.Process = process;
            session.Mode = mode;
            session.Paused = session.Stopping = session.CommandPending = false;
            session.Rate = "--";
            session.Clicks = "0";
            session.Interval = "最近连点：--";
            session.State = mode == "probe" ? "正在检测" : "正在启动";
            if (mode == "probe") session.Probe = "正在检查连接…";
            ShowMessage(mode == "probe" ? "正在检查模拟器连接和识别设备…" :
                mode == "hayabusa-debug" ? "场景调试：仅观察所选实例，不发送战斗动作…" :
                "正在连接模拟器并初始化识别…", session);
            RefreshSession(session);
            _ = ObserveWorkerAsync(session, process);
        }
        catch (Exception error) when (error is Win32Exception or IOException or InvalidOperationException)
        {
            if (mode == "probe") session.Probe = "启动失败：" + error.Message;
            ShowMessage("启动失败：" + error.Message, session);
            RefreshSession(session);
        }
    }

    private async Task ObserveWorkerAsync(WorkerSession session, Process process)
    {
        async Task ReadAsync(StreamReader reader, bool isError)
        {
            while (await reader.ReadLineAsync() is { } line)
            {
                DispatcherQueue.TryEnqueue(() =>
                {
                    if (!ReferenceEquals(session.Process, process)) return;
                    if (isError) AddLog("运行组件：" + line, session);
                    else HandleWorkerLine(session, line);
                });
            }
        }
        try
        {
            var output = ReadAsync(process.StandardOutput, false);
            var errors = ReadAsync(process.StandardError, true);
            await process.WaitForExitAsync();
            await Task.WhenAll(output, errors);
            int code = process.ExitCode;
            DispatcherQueue.TryEnqueue(() => WorkerExited(session, process, code));
        }
        catch (Exception error) when (error is IOException or InvalidOperationException)
        {
            DispatcherQueue.TryEnqueue(() =>
            {
                if (ReferenceEquals(session.Process, process)) ShowMessage("运行组件已中断：" + error.Message, session);
                WorkerExited(session, process, -1);
            });
        }
    }

    private void HandleWorkerLine(WorkerSession session, string line)
    {
        if (!line.TrimStart().StartsWith('{')) return;
        try
        {
            using var json = JsonDocument.Parse(line);
            var root = json.RootElement;
            string? kind = root.GetProperty("event").GetString();
            var value = root.GetProperty("value");
            switch (kind)
            {
                case "backend":
                    session.Device = "识别设备：" + value.GetString();
                    break;
                case "status":
                    if (session.State == "正在启动") session.State = "运行中";
                    ShowMessage(value.GetString() ?? "", session);
                    break;
                case "state":
                    bool wasPaused = session.Paused;
                    session.Paused = value.GetString() == "paused";
                    if (!session.Stopping)
                        session.State = session.Paused ? "已暂停" : "运行中";
                    if (session.Paused && !wasPaused) ShowMessage("已暂停，继续后会重新识别当前画面。", session);
                    else if (!session.Paused && wasPaused) ShowMessage("正在继续，重新识别当前画面…", session);
                    break;
                case "metrics":
                    session.Rate = value.GetProperty("fps").GetDouble().ToString("F1");
                    session.Clicks = value.TryGetProperty("clicks", out var clicks)
                        ? clicks.GetInt32().ToString() : "—";
                    if (value.TryGetProperty("join_interval_ms", out var interval) && interval.ValueKind == JsonValueKind.Number)
                        session.Interval = "最近连点：" + interval.GetDouble().ToString("F0") + " 毫秒";
                    if (ReferenceEquals(SelectedSession, session)) RenderMetrics(session);
                    return;
                case "kill":
                    session.Completed += value.GetInt32();
                    AddLog("已确认结算，累计 " + session.Completed + " 次。", session);
                    break;
                case "probe":
                    session.Device = "识别设备：" + value.GetProperty("backend").GetString();
                    session.Probe = "连接成功：实例画面 " + value.GetProperty("width").GetInt32() + " × " +
                        value.GetProperty("height").GetInt32() + "，" + value.GetProperty("page").GetString() + "。";
                    ShowMessage(session.Probe, session);
                    break;
                case "error":
                    if (session.Mode == "probe") session.Probe = "连接检查失败：" + value.GetString();
                    ShowMessage(session.Mode == "probe" ? session.Probe : "任务异常：" + value.GetString(), session);
                    break;
                default:
                    return;
            }
            RefreshSession(session);
        }
        catch (Exception error) when (error is JsonException or InvalidOperationException or KeyNotFoundException)
        {
            AddLog("无法读取运行状态：" + error.Message, session);
        }
    }

    private void WorkerExited(WorkerSession session, Process process, int exitCode)
    {
        if (!ReferenceEquals(session.Process, process)) return;
        session.Process = null;
        process.Dispose();
        session.Stopping = session.Paused = session.CommandPending = false;
        session.State = exitCode == 0 ? "未运行" : "异常退出";
        if (exitCode != 0)
        {
            if (session.Mode == "probe") session.Probe = "连接检查组件已退出，代码：" + exitCode;
            ShowMessage("运行组件已退出，代码：" + exitCode, session);
        }
        else if (session.Mode is "run" or "hayabusa-debug") ShowMessage("任务已结束。", session);
        RefreshSession(session);
    }

    private async void Pause_Click(object sender, RoutedEventArgs e)
    {
        if (SelectedSession is not { Process: { } process, Stopping: false, CommandPending: false } session ||
            session.Mode is not ("run" or "hayabusa-debug")) return;
        await SendCommandAsync(session, process, session.Paused ? "resume" : "pause");
    }

    private async void Stop_Click(object sender, RoutedEventArgs e)
    {
        if (SelectedSession is not { Process: { } process, Stopping: false, CommandPending: false } session ||
            session.Mode is not ("run" or "hayabusa-debug")) return;
        session.Stopping = true;
        session.State = "正在停止";
        ShowMessage("正在停止任务…", session);
        await SendCommandAsync(session, process, "stop");
    }

    private async Task SendCommandAsync(WorkerSession session, Process process, string command)
    {
        session.CommandPending = true;
        RefreshSession(session);
        try
        {
            if (!process.HasExited)
            {
                await process.StandardInput.WriteLineAsync(JsonSerializer.Serialize(new { command }));
                await process.StandardInput.FlushAsync();
            }
        }
        catch (Exception error) when (error is IOException or InvalidOperationException)
        {
            if (ReferenceEquals(session.Process, process))
            {
                session.Stopping = false;
                session.State = session.Paused ? "已暂停" : "运行中";
                ShowMessage("无法控制任务：" + error.Message, session);
            }
        }
        finally
        {
            if (ReferenceEquals(session.Process, process))
            {
                session.CommandPending = false;
                RefreshSession(session);
            }
        }
    }

    private void RefreshSession(WorkerSession session)
    {
        if (InstanceBox.ItemsSource is IEnumerable<InstanceOption> instances)
            foreach (var item in instances.Where(item => ReferenceEquals(item.Session, session)))
                item.Refresh();
        if (ReferenceEquals(SelectedSession, session)) RenderSelectedSession();
        UpdateControls();
    }

    private void RenderMetrics(WorkerSession? session)
    {
        RateText.Text = session?.Rate ?? "--";
        ClicksText.Text = session?.Clicks ?? "0";
        ActualIntervalText.Text = session?.Interval ?? "最近连点：--";
        CountText.Text = (session?.Completed ?? _restoredCount).ToString();
    }

    private void RenderSelectedSession()
    {
        var session = SelectedSession;
        bool debug = session?.Mode == "hayabusa-debug";
        StatusText.Text = debug && session!.Process is not null
            ? "当前实例正在使用场景调试。" : session?.Status ?? "请先选择一个 MuMu 实例。";
        DebugStatusText.Text = debug ? session!.Status : "调试只观察当前场景，最长 10 分钟。";
        StatusBadge.Text = "●  " + (session?.State ?? "未运行");
        DeviceText.Text = session?.Device ?? "识别设备：尚未检测";
        ProbeResultText.Text = session?.Probe ?? "";
        PauseButton.Content = session?.Paused == true ? "继续" : "暂停";
        DebugPauseButton.Content = session?.Paused == true ? "继续调试" : "暂停调试";
        RenderMetrics(session);
        RenderLog(session);
    }

    private void UpdateControls()
    {
        var session = SelectedSession;
        bool available = !_exiting && !_scanning && !VideoDebugBusy;
        bool editable = available && session is { Process: null };
        bool controllable = !_exiting && session is { Process: not null, Stopping: false, CommandPending: false };
        StartButton.IsEnabled = DebugStartButton.IsEnabled = ProbeButton.IsEnabled = SaveButton.IsEnabled = editable;
        PauseButton.IsEnabled = StopButton.IsEnabled = controllable && session?.Mode == "run";
        DebugPauseButton.IsEnabled = DebugStopButton.IsEnabled = controllable && session?.Mode == "hayabusa-debug";
        FolderButton.IsEnabled = FolderBox.IsEnabled = available && !AnyWorker;
        InstanceBox.IsEnabled = ScanButton.IsEnabled = RefreshInstancesButton.IsEnabled = available;
        HideGameWindowSwitch.IsEnabled = JoinIntervalBox.IsEnabled = editable;
        ActivityEnabledSwitch.IsEnabled = WeekendEnabledSwitch.IsEnabled = editable;
        XumiEnabledSwitch.IsEnabled = AbyssEnabledSwitch.IsEnabled = DailyEnabledSwitch.IsEnabled = editable;
        WeekendModeBox.IsEnabled = editable && WeekendEnabledSwitch.IsOn;
        ActivityModeBox.IsEnabled = editable && ActivityEnabledSwitch.IsOn;
        DailyOptionsControl.IsEnabled = editable && DailyEnabledSwitch.IsOn;
        UpdateVideoDebugAvailability(AnyWorker || _scanning);

        int count = _sessions.Values.Count(item => item.Process is not null);
        int paused = _sessions.Values.Count(item => item.Process is not null && item.Paused);
        FleetText.Text = count == 0 ? "MuMu 多开" : paused == 0
            ? $"MuMu 多开 · {count} 个活跃" : $"MuMu 多开 · {count} 个活跃 · {paused} 个已暂停";
    }

    private void DailyEnabled_Toggled(object sender, RoutedEventArgs e)
    {
        if (DailyOptionsControl is not null)
            DailyOptionsControl.IsEnabled = DailyEnabledSwitch.IsOn && SelectedSession is { Process: null } && !_scanning && !VideoDebugBusy;
    }

    private void BountyTask_Changed(object sender, RoutedEventArgs e) => UpdateBountyOptions();

    private void UpdateBountyOptions()
    {
        bool regularBounty = BountyTaskOption?.IsChecked == true;
        if (BountyGradeSlider is not null)
            BountyGradeSlider.IsEnabled = regularBounty;
        if (BountyInviteOptionsControl is not null)
            BountyInviteOptionsControl.IsEnabled = regularBounty || SBountyTaskOption?.IsChecked == true;
    }

    private void PetFeedTask_Changed(object sender, RoutedEventArgs e)
    {
        if (PetFeedRewardBox is not null)
            PetFeedRewardBox.IsEnabled = PetFeedTaskOption.IsChecked == true;
    }

    private void BountyGrade_ValueChanged(object sender, Microsoft.UI.Xaml.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        if (BountyGradeText is not null)
            BountyGradeText.Text = BountyGrades[(int)Math.Round(e.NewValue)];
    }

    private void ActivityEnabled_Toggled(object sender, RoutedEventArgs e)
    {
        if (ActivityModeBox is not null)
            ActivityModeBox.IsEnabled = ActivityEnabledSwitch.IsOn && SelectedSession is { Process: null } && !_scanning && !VideoDebugBusy;
    }

    private void ActivityTabs_SelectionChanged(SelectorBar sender, SelectorBarSelectionChangedEventArgs args)
    {
        IntelligenceOptions.Visibility = sender.SelectedItem == IntelligenceTab ? Visibility.Visible : Visibility.Collapsed;
        XumiOptions.Visibility = sender.SelectedItem == XumiTab ? Visibility.Visible : Visibility.Collapsed;
    }

    private void WeekendEnabled_Toggled(object sender, RoutedEventArgs e)
    {
        if (WeekendModeBox is not null)
            WeekendModeBox.IsEnabled = WeekendEnabledSwitch.IsOn && SelectedSession is { Process: null } && !_scanning && !VideoDebugBusy;
    }

    private async void Browse_Click(object sender, RoutedEventArgs e)
    {
        var picker = new FolderPicker();
        picker.FileTypeFilter.Add("*");
        WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
        var folder = await picker.PickSingleFolderAsync();
        if (folder is not null)
        {
            FolderBox.Text = folder.Path;
            await ScanInstancesAsync();
        }
    }

    private async void Save_Click(object sender, RoutedEventArgs e)
    {
        if (!string.Equals(FolderBox.Text.Trim(), _scannedFolder, StringComparison.OrdinalIgnoreCase))
        {
            await ScanInstancesAsync();
            if (!string.Equals(FolderBox.Text.Trim(), _scannedFolder, StringComparison.OrdinalIgnoreCase)) return;
        }
        if (TrySettings(out var settings, requireRunning: false) && PersistSettings(settings))
            ShowMessage("设备设置已保存。");
    }

    private void TaskNav_Click(object sender, RoutedEventArgs e) => ShowPage(true);
    private void SettingsNav_Click(object sender, RoutedEventArgs e) => ShowPage(false);

    private void ShowPage(bool task)
    {
        TaskPage.Visibility = task ? Visibility.Visible : Visibility.Collapsed;
        SettingsPage.Visibility = task ? Visibility.Collapsed : Visibility.Visible;
        VideoPage.Visibility = Visibility.Collapsed;
        VideoNav.Background = new SolidColorBrush(Windows.UI.Color.FromArgb(0, 0, 0, 0));
        TaskNav.Background = new SolidColorBrush(task ? Windows.UI.Color.FromArgb(255, 38, 52, 81) : Windows.UI.Color.FromArgb(0, 0, 0, 0));
        SettingsNav.Background = new SolidColorBrush(task ? Windows.UI.Color.FromArgb(0, 0, 0, 0) : Windows.UI.Color.FromArgb(255, 38, 52, 81));
    }

    private void ShowMessage(string message, WorkerSession? session = null)
    {
        session ??= SelectedSession;
        if (session is not null) session.Status = message;
        if (ReferenceEquals(session, SelectedSession))
        {
            if (session?.Mode == "hayabusa-debug") DebugStatusText.Text = message;
            else StatusText.Text = message;
        }
        AddLog(message, session);
    }

    private void AddLog(string message, WorkerSession? session = null)
    {
        session ??= SelectedSession;
        var log = session?.Log ?? _log;
        log.Enqueue(DateTime.Now.ToString("HH:mm:ss") + "  " + message);
        while (log.Count > 100) log.Dequeue();
        if (ReferenceEquals(SelectedSession, session)) RenderLog(session);
    }

    private void RenderLog(WorkerSession? session)
    {
        LogBox.Text = string.Join(Environment.NewLine, session?.Log ?? _log);
        LogBox.SelectionStart = LogBox.Text.Length;
    }

    private void Window_Closed(object sender, WindowEventArgs args)
    {
        DisposeTrayIntegration();
        CloseVideoDebug(updateControls: false);
        foreach (var session in _sessions.Values)
        {
            try { session.Process?.StandardInput.Close(); } // EOF requests a clean Worker stop.
            catch (Exception error) when (error is IOException or InvalidOperationException) { }
        }
    }
}

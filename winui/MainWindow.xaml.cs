using System.Diagnostics;
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
    private readonly string _settingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Ninja3Assistant", "settings.json");
    private readonly Queue<string> _log = new();
    private Process? _worker;
    private string _mode = "";
    private bool _paused;
    private bool _stopping;
    private bool _scanning;
    private int _completed;
    private int? _savedIndex;
    private string? _savedName;
    private string? _savedFolder;
    private string? _scannedFolder;

    private sealed class Settings
    {
        public string Folder { get; set; } = @"D:\Program Files\Netease\MuMu";
        public int? Index { get; set; }
        public string? InstanceName { get; set; }
        public int JoinIntervalMs { get; set; } = 30;
        public bool ActivityEnabled { get; set; }
        public string ActivityMode { get; set; } = "daily";
        public bool WeekendEnabled { get; set; }
        public string WeekendMode { get; set; } = "daily";
        public bool HideGameWindow { get; set; }
        public bool DailyEnabled { get; set; }
        public string[] DailyTasks { get; set; } = ["monthly", "limited", "blessing", "weapon", "curse", "bounty_shop", "daily_shop", "village", "house", "bounty", "bounty_s", "rank"];
        public string BountyGrade { get; set; } = "D";
        public bool CloseToTray { get; set; } = true;
        public bool RememberCloseAction { get; set; }
    }

    private sealed class InstanceOption
    {
        public required int Index { get; init; }
        public required string Name { get; init; }
        public required bool Started { get; init; }
        public string DisplayName => Name + (Started ? "  ·  运行中" : "  ·  未启动");
    }

    public MainWindow()
    {
        InitializeComponent();
        InitializeVideoDebug();
        AppWindow.Resize(new SizeInt32(1080, 760));
        Closed += Window_Closed;
        LoadSettings();
        InitializeTrayIntegration();
        AddLog("准备就绪。");
        RestoreSessionCount();
    }

    private void RestoreSessionCount()
    {
        string[] args = Environment.GetCommandLineArgs();
        int index = Array.IndexOf(args, "--resume-settlements");
        if (index < 0 || index + 1 >= args.Length || !int.TryParse(args[index + 1], out int count) || count < 0)
            return;
        _completed = count;
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
        FolderBox.Text = settings.Folder;
        JoinIntervalBox.Value = settings.JoinIntervalMs is >= 20 and <= 1000 ? settings.JoinIntervalMs : 30;
        ActivityEnabledSwitch.IsOn = settings.ActivityEnabled;
        ActivityModeBox.SelectedIndex = settings.ActivityMode == "continuous" ? 1 : 0;
        ActivityModeBox.IsEnabled = settings.ActivityEnabled;
        WeekendEnabledSwitch.IsOn = settings.WeekendEnabled;
        WeekendModeBox.SelectedIndex = settings.WeekendMode == "25" ? 1 : 0;
        WeekendModeBox.IsEnabled = settings.WeekendEnabled;
        HideGameWindowSwitch.IsOn = settings.HideGameWindow;
        DailyEnabledSwitch.IsOn = settings.DailyEnabled;
        foreach (var option in DailyOptionsPanel.Children.OfType<CheckBox>())
            option.IsChecked = (settings.DailyTasks ?? []).Contains((string)option.Tag);
        DailyOptionsControl.IsEnabled = settings.DailyEnabled;
        BountyGradeSlider.Value = Math.Max(0, Array.IndexOf(BountyGrades, settings.BountyGrade));
        BountyGradeSlider.IsEnabled = BountyTaskOption.IsChecked == true;
        LoadCloseBehaviorSettings(settings);
        _savedIndex = settings.Index;
        _savedName = settings.InstanceName;
        _savedFolder = settings.Folder;
    }

    private bool TrySettings(out Settings settings, bool requireRunning = true)
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
        settings.Index = instance.Index;
        settings.InstanceName = instance.Name;
        if (double.IsNaN(JoinIntervalBox.Value) || JoinIntervalBox.Value < 20 ||
            JoinIntervalBox.Value > 1000 || JoinIntervalBox.Value != Math.Truncate(JoinIntervalBox.Value))
        {
            ShowMessage("抢队连点间隔请输入 20 到 1000 之间的整数毫秒。");
            return false;
        }
        settings.JoinIntervalMs = (int)JoinIntervalBox.Value;
        settings.ActivityEnabled = ActivityEnabledSwitch.IsOn;
        settings.ActivityMode = ActivityModeBox.SelectedIndex == 1 ? "continuous" : "daily";
        settings.WeekendEnabled = WeekendEnabledSwitch.IsOn;
        settings.WeekendMode = WeekendModeBox.SelectedIndex == 1 ? "25" : "daily";
        settings.HideGameWindow = HideGameWindowSwitch.IsOn;
        settings.DailyEnabled = DailyEnabledSwitch.IsOn;
        settings.DailyTasks = DailyOptionsPanel.Children.OfType<CheckBox>()
            .Where(option => option.IsChecked == true).Select(option => (string)option.Tag).ToArray();
        settings.BountyGrade = BountyGrades[(int)Math.Round(BountyGradeSlider.Value)];
        settings.CloseToTray = _closeToTray;
        settings.RememberCloseAction = _rememberCloseAction;
        return true;
    }

    private bool PersistSettings(Settings settings)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_settingsPath)!);
            string temp = _settingsPath + ".tmp";
            File.WriteAllText(temp, JsonSerializer.Serialize(settings, new JsonSerializerOptions { WriteIndented = true }));
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

    private void Instance_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        DeviceText.Text = "识别设备：尚未检测";
        ProbeResultText.Text = "";
        if (InstanceBox.SelectedItem is InstanceOption instance)
            ScanText.Text = instance.Started ? "已选择正在运行的实例。" : "所选实例尚未启动，启动任务时会自动打开。";
    }
    private async void Scan_Click(object sender, RoutedEventArgs e) => await ScanInstancesAsync();

    private async Task ScanInstancesAsync()
    {
        if (_exiting || _scanning || _worker is not null || VideoDebugBusy) return;
        string folder = FolderBox.Text.Trim();
        string manager = Path.Combine(folder, "nx_main", "MuMuManager.exe");
        var previous = InstanceBox.SelectedItem as InstanceOption;
        bool sameFolder = string.Equals(folder, _scannedFolder, StringComparison.OrdinalIgnoreCase);
        int? preferredIndex = sameFolder ? previous?.Index ?? _savedIndex :
            string.Equals(folder, _savedFolder, StringComparison.OrdinalIgnoreCase) ? _savedIndex : null;
        string? preferredName = sameFolder ? previous?.Name ?? _savedName :
            string.Equals(folder, _savedFolder, StringComparison.OrdinalIgnoreCase) ? _savedName : null;
        _scanning = true;
        SetBusy(false);
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
                instances.Add(new InstanceOption { Index = index, Name = name, Started = started });
            }
            instances.Sort((a, b) => a.Index.CompareTo(b.Index));
            _scannedFolder = folder;
            InstanceBox.ItemsSource = instances;
            InstanceBox.SelectedItem = instances.FirstOrDefault(item => item.Index == preferredIndex &&
                (preferredName is null || item.Name == preferredName));
            if (InstanceBox.SelectedItem is null && instances.Count == 1) InstanceBox.SelectedItem = instances[0];
            int runningCount = instances.Count(item => item.Started);
            ScanText.Text = instances.Count == 0 ? "未发现模拟器实例，请先在 MuMu 多开器中创建。" :
                $"已发现 {instances.Count} 个实例，其中 {runningCount} 个运行中。" +
                (InstanceBox.SelectedItem is null ? "请选择要控制的实例。" : "");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException or
                                      InvalidOperationException or TimeoutException or System.ComponentModel.Win32Exception)
        {
            _scannedFolder = null;
            InstanceBox.ItemsSource = null;
            ScanText.Text = "扫描失败：" + error.Message;
            AddLog(ScanText.Text);
        }
        finally
        {
            _scanning = false;
            SetBusy(false);
        }
    }

    private async void Start_Click(object sender, RoutedEventArgs e) => await StartWorkerAsync("run");
    private async void DebugStart_Click(object sender, RoutedEventArgs e) => await StartWorkerAsync("hayabusa-debug");
    private async void Probe_Click(object sender, RoutedEventArgs e) => await StartWorkerAsync("probe");

    private async Task StartWorkerAsync(string mode)
    {
        if (_exiting || _worker is not null || _scanning || VideoDebugBusy) return;
        if (!string.Equals(FolderBox.Text.Trim(), _scannedFolder, StringComparison.OrdinalIgnoreCase))
        {
            await ScanInstancesAsync();
            if (_exiting || !string.Equals(FolderBox.Text.Trim(), _scannedFolder, StringComparison.OrdinalIgnoreCase)) return;
        }
        if (!TrySettings(out var settings, requireRunning: mode != "run") || !PersistSettings(settings))
            return;
        if (mode == "run" && settings.DailyEnabled && settings.DailyTasks.Length == 0)
        {
            ShowMessage("请至少选择一项每日清理任务。");
            return;
        }
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
            if (mode == "run" && settings.DailyEnabled)
            {
                info.ArgumentList.Add("--daily-tasks");
                foreach (string task in settings.DailyTasks) info.ArgumentList.Add(task);
                if (settings.DailyTasks.Contains("bounty"))
                {
                    info.ArgumentList.Add("--bounty-grade");
                    info.ArgumentList.Add(settings.BountyGrade);
                }
            }
            var process = Process.Start(info) ?? throw new InvalidOperationException("无法启动运行组件");
            _worker = process;
            _mode = mode;
            _paused = false;
            _stopping = false;
            RateText.Text = "--";
            ClicksText.Text = "0";
            ActualIntervalText.Text = "最近连点：--";
            PauseButton.Content = "暂停";
            DebugPauseButton.Content = "暂停调试";
            SetBusy(true);
            StatusBadge.Text = mode == "probe" ? "●  正在检测" : "●  正在启动";
            if (mode == "probe") ProbeResultText.Text = "正在检查连接…";
            ShowMessage(mode == "probe" ? "正在检查模拟器连接和识别设备…" :
                mode == "hayabusa-debug" ? "场景调试：仅观察所选实例，不发送战斗动作…" :
                "正在连接模拟器并初始化识别…");
            _ = ObserveWorkerAsync(process);
        }
        catch (Exception error) when (error is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
            _worker = null;
            SetBusy(false);
            if (mode == "probe") ProbeResultText.Text = "启动失败：" + error.Message;
            ShowMessage("启动失败：" + error.Message);
        }
    }

    private async Task ObserveWorkerAsync(Process process)
    {
        async Task ReadAsync(StreamReader reader, bool isError)
        {
            string? line;
            while ((line = await reader.ReadLineAsync()) is not null)
            {
                string captured = line;
                DispatcherQueue.TryEnqueue(() =>
                {
                    if (!ReferenceEquals(_worker, process)) return;
                    if (isError) AddLog("运行组件：" + captured);
                    else HandleWorkerLine(captured);
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
            DispatcherQueue.TryEnqueue(() => WorkerExited(process, code));
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or ObjectDisposedException)
        {
            DispatcherQueue.TryEnqueue(() =>
            {
                if (ReferenceEquals(_worker, process)) ShowMessage("运行组件已中断：" + error.Message);
                WorkerExited(process, -1);
            });
        }
    }

    private void HandleWorkerLine(string line)
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
                    DeviceText.Text = "识别设备：" + value.GetString();
                    break;
                case "status":
                    ShowMessage(value.GetString() ?? "");
                    break;
                case "state":
                    bool wasPaused = _paused;
                    _paused = value.GetString() == "paused";
                    PauseButton.Content = _paused ? "继续" : "暂停";
                    DebugPauseButton.Content = _paused ? "继续调试" : "暂停调试";
                    StatusBadge.Text = _paused ? "●  已暂停" : "●  运行中";
                    if (_paused && !wasPaused) ShowMessage("已暂停；继续后会重新识别当前画面。");
                    else if (!_paused && wasPaused) ShowMessage("正在继续，重新识别当前画面…");
                    break;
                case "metrics":
                    RateText.Text = value.GetProperty("fps").GetDouble().ToString("F1");
                    ClicksText.Text = value.TryGetProperty("clicks", out var clicks)
                        ? clicks.GetInt32().ToString() : "—";
                    if (value.TryGetProperty("join_interval_ms", out var interval) && interval.ValueKind == JsonValueKind.Number)
                        ActualIntervalText.Text = "最近连点：" + interval.GetDouble().ToString("F0") + " 毫秒";
                    break;
                case "kill":
                    _completed += value.GetInt32();
                    CountText.Text = _completed.ToString();
                    AddLog("已确认结算，累计 " + _completed + " 次。");
                    break;
                case "probe":
                    DeviceText.Text = "识别设备：" + value.GetProperty("backend").GetString();
                    ProbeResultText.Text = "连接成功：实例画面 " + value.GetProperty("width").GetInt32() + " × " +
                        value.GetProperty("height").GetInt32() + "，" + value.GetProperty("page").GetString() + "。";
                    ShowMessage(ProbeResultText.Text);
                    break;
                case "error":
                    if (_mode == "probe") ProbeResultText.Text = "连接检查失败：" + value.GetString();
                    ShowMessage(_mode == "probe" ? ProbeResultText.Text : "任务异常：" + value.GetString());
                    break;
            }
        }
        catch (Exception error) when (error is JsonException or InvalidOperationException or KeyNotFoundException)
        {
            AddLog("无法读取运行状态：" + error.Message);
        }
    }

    private void WorkerExited(Process process, int exitCode)
    {
        if (!ReferenceEquals(_worker, process)) return;
        _worker = null;
        process.Dispose();
        SetBusy(false);
        StatusBadge.Text = "●  未运行";
        if (exitCode != 0)
        {
            if (_mode == "probe") ProbeResultText.Text = "连接检查组件已退出，代码：" + exitCode;
            ShowMessage("运行组件已退出，代码：" + exitCode);
        }
        else if (_mode is "run" or "hayabusa-debug") AddLog("任务已结束。");
    }

    private async void Pause_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _mode is not ("run" or "hayabusa-debug") || _stopping) return;
        await SendCommandAsync(_paused ? "resume" : "pause");
    }

    private async void Stop_Click(object sender, RoutedEventArgs e)
    {
        if (_worker is null || _mode is not ("run" or "hayabusa-debug")) return;
        _stopping = true;
        PauseButton.IsEnabled = false;
        StopButton.IsEnabled = false;
        DebugPauseButton.IsEnabled = false;
        DebugStopButton.IsEnabled = false;
        StatusBadge.Text = "●  正在停止";
        ShowMessage("正在停止任务…");
        await SendCommandAsync("stop");
    }

    private async Task SendCommandAsync(string command)
    {
        try
        {
            if (_worker is not null && !_worker.HasExited)
            {
                await _worker.StandardInput.WriteLineAsync(JsonSerializer.Serialize(new { command }));
                await _worker.StandardInput.FlushAsync();
            }
        }
        catch (Exception error) when (error is IOException or InvalidOperationException)
        {
            ShowMessage("无法控制任务：" + error.Message);
        }
    }

    private void SetBusy(bool busy)
    {
        bool available = !_exiting && !busy && !_scanning && !VideoDebugBusy;
        StartButton.IsEnabled = available;
        PauseButton.IsEnabled = busy && _mode == "run";
        StopButton.IsEnabled = busy && _mode == "run";
        DebugStartButton.IsEnabled = available;
        DebugPauseButton.IsEnabled = busy && _mode == "hayabusa-debug";
        DebugStopButton.IsEnabled = busy && _mode == "hayabusa-debug";
        ProbeButton.IsEnabled = available;
        SaveButton.IsEnabled = available;
        FolderButton.IsEnabled = available;
        FolderBox.IsEnabled = available;
        HideGameWindowSwitch.IsEnabled = available;
        InstanceBox.IsEnabled = available;
        ScanButton.IsEnabled = available;
        JoinIntervalBox.IsEnabled = available;
        ActivityEnabledSwitch.IsEnabled = available;
        WeekendEnabledSwitch.IsEnabled = available;
        WeekendModeBox.IsEnabled = available && WeekendEnabledSwitch.IsOn;
        ActivityModeBox.IsEnabled = available && ActivityEnabledSwitch.IsOn;
        DailyEnabledSwitch.IsEnabled = available;
        DailyOptionsControl.IsEnabled = available && DailyEnabledSwitch.IsOn;
        UpdateVideoDebugAvailability(busy || _scanning);
    }

    private void DailyEnabled_Toggled(object sender, RoutedEventArgs e)
    {
        if (DailyOptionsControl is not null)
            DailyOptionsControl.IsEnabled = DailyEnabledSwitch.IsOn && _worker is null && !_scanning && !VideoDebugBusy;
    }

    private void BountyTask_Changed(object sender, RoutedEventArgs e)
    {
        if (BountyGradeSlider is not null)
            BountyGradeSlider.IsEnabled = BountyTaskOption.IsChecked == true;
    }

    private void BountyGrade_ValueChanged(object sender, Microsoft.UI.Xaml.Controls.Primitives.RangeBaseValueChangedEventArgs e)
    {
        if (BountyGradeText is not null)
            BountyGradeText.Text = BountyGrades[(int)Math.Round(e.NewValue)];
    }

    private void ActivityEnabled_Toggled(object sender, RoutedEventArgs e)
    {
        if (ActivityModeBox is not null)
            ActivityModeBox.IsEnabled = ActivityEnabledSwitch.IsOn && _worker is null && !_scanning && !VideoDebugBusy;
    }

    private void WeekendEnabled_Toggled(object sender, RoutedEventArgs e)
    {
        if (WeekendModeBox is not null)
            WeekendModeBox.IsEnabled = WeekendEnabledSwitch.IsOn && _worker is null && !_scanning && !VideoDebugBusy;
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

    private void ShowMessage(string message)
    {
        if (_mode == "hayabusa-debug") DebugStatusText.Text = message;
        else StatusText.Text = message;
        AddLog(message);
    }

    private void AddLog(string message)
    {
        _log.Enqueue(DateTime.Now.ToString("HH:mm:ss") + "  " + message);
        while (_log.Count > 100) _log.Dequeue();
        LogBox.Text = string.Join(Environment.NewLine, _log);
        LogBox.SelectionStart = LogBox.Text.Length;
    }

    private void Window_Closed(object sender, WindowEventArgs args)
    {
        DisposeTrayIntegration();
        CloseVideoDebug(updateControls: false);
        var process = _worker;
        if (process is null) return;
        try
        {
            if (!process.HasExited)
            {
                process.StandardInput.WriteLine("{\"command\":\"stop\"}");
                process.StandardInput.Flush();
                process.StandardInput.Close();
                if (!process.WaitForExit(3000)) process.Kill(true);
            }
        }
        catch (Exception error) when (error is IOException or InvalidOperationException or System.ComponentModel.Win32Exception) { }
    }
}

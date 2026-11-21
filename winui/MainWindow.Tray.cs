using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Ninja3Assistant;

public sealed partial class MainWindow
{
    private NativeTrayIcon? _trayIcon;
    private bool _closeToTray = true;
    private bool _rememberCloseAction;
    private bool _loadingCloseBehavior;
    private bool _hiddenInTray;
    private bool _exiting;
    private bool _allowWindowClose;
    private ContentDialog? _closeDialog;

    private void InitializeTrayIntegration()
    {
        string iconPath = Path.Combine(AppContext.BaseDirectory, "Assets", "Ninja3Icon.ico");
        try
        {
            if (File.Exists(iconPath)) AppWindow.SetIcon(iconPath);
        }
        catch (Exception error) when (error is IOException or ArgumentException or COMException)
        {
            AddLog("窗口图标加载失败，将使用默认图标：" + error.Message);
        }
        try
        {
            _trayIcon = new NativeTrayIcon(WinRT.Interop.WindowNative.GetWindowHandle(this), iconPath);
            _trayIcon.RestoreRequested += () => DispatcherQueue.TryEnqueue(RestoreFromTray);
            _trayIcon.ExitRequested += () => DispatcherQueue.TryEnqueue(async () => await ExitApplicationAsync());
            _trayIcon.Unavailable += () => DispatcherQueue.TryEnqueue(() =>
            {
                RestoreFromTray();
                ShowMessage("托盘暂不可用，已保留主窗口。");
            });
        }
        catch (Win32Exception error)
        {
            AddLog("托盘初始化失败，将保留主窗口：" + error.Message);
        }
        AppWindow.Closing += AppWindow_Closing;
        AppWindow.Changed += AppWindow_Changed;
    }

    private void LoadCloseBehaviorSettings(Settings settings)
    {
        SetCloseBehavior(settings.CloseToTray, settings.RememberCloseAction);
    }

    private void SetCloseBehavior(bool closeToTray, bool remember)
    {
        _loadingCloseBehavior = true;
        _closeToTray = closeToTray;
        _rememberCloseAction = remember;
        CloseToTraySwitch.IsOn = closeToTray;
        RememberCloseActionSwitch.IsOn = remember;
        _loadingCloseBehavior = false;
    }

    private bool SaveCloseBehavior()
    {
        try
        {
            var settings = File.Exists(_settingsPath)
                ? JsonNode.Parse(File.ReadAllText(_settingsPath)) as JsonObject
                    ?? throw new JsonException("设置内容不是对象。")
                : new JsonObject();
            settings[nameof(Settings.CloseToTray)] = _closeToTray;
            settings[nameof(Settings.RememberCloseAction)] = _rememberCloseAction;
            Directory.CreateDirectory(Path.GetDirectoryName(_settingsPath)!);
            string temp = _settingsPath + ".tmp";
            File.WriteAllText(temp, settings.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            File.Move(temp, _settingsPath, true);
            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException)
        {
            ShowMessage("保存关闭偏好失败：" + error.Message);
            return false;
        }
    }

    private void CloseBehavior_Changed(object sender, RoutedEventArgs e)
    {
        if (_loadingCloseBehavior || CloseToTraySwitch is null || RememberCloseActionSwitch is null) return;
        _closeToTray = CloseToTraySwitch.IsOn;
        _rememberCloseAction = RememberCloseActionSwitch.IsOn;
        SaveCloseBehavior();
    }

    private void ResetCloseBehavior_Click(object sender, RoutedEventArgs e)
    {
        SetCloseBehavior(true, false);
        if (SaveCloseBehavior()) ShowMessage("已恢复每次关闭时询问，默认最小化到托盘。");
    }

    private void AppWindow_Changed(AppWindow sender, AppWindowChangedEventArgs args)
    {
        if (!_exiting && !_hiddenInTray && sender.Presenter is OverlappedPresenter presenter &&
            presenter.State == OverlappedPresenterState.Minimized)
            HideToTray();
    }

    private bool HideToTray()
    {
        if (_trayIcon?.EnsureRegistered() != true)
        {
            RestoreFromTray();
            ShowMessage("托盘暂不可用，主窗口将保持显示。");
            return false;
        }
        _closeDialog?.Hide();
        _hiddenInTray = true;
        AppWindow.Hide();
        return true;
    }

    private void RestoreFromTray()
    {
        if (_allowWindowClose) return;
        _hiddenInTray = false;
        if (AppWindow.Presenter is OverlappedPresenter presenter &&
            presenter.State == OverlappedPresenterState.Minimized)
            presenter.Restore();
        AppWindow.Show();
        Activate();
    }

    private async void AppWindow_Closing(AppWindow sender, AppWindowClosingEventArgs args)
    {
        if (_allowWindowClose) return;
        args.Cancel = true;
        if (_exiting || _closeDialog is not null) return;
        if (_rememberCloseAction)
        {
            if (_closeToTray) HideToTray();
            else await ExitApplicationAsync();
            return;
        }
        if (Content is not FrameworkElement root || root.XamlRoot is null) return;
        var remember = new CheckBox { Content = "记住本次选择，以后不再询问" };
        var content = new StackPanel { Spacing = 16 };
        content.Children.Add(new TextBlock
        {
            Text = "最小化到托盘后，任务会继续运行。退出助手会停止任务。",
            TextWrapping = TextWrapping.Wrap
        });
        content.Children.Add(remember);
        _closeDialog = new ContentDialog
        {
            XamlRoot = root.XamlRoot,
            RequestedTheme = root.ActualTheme,
            Title = "关闭助手",
            Content = content,
            PrimaryButtonText = "最小化到托盘",
            SecondaryButtonText = "退出助手",
            CloseButtonText = "取消",
            DefaultButton = _closeToTray ? ContentDialogButton.Primary : ContentDialogButton.Secondary
        };
        ContentDialogResult result;
        try
        {
            result = await _closeDialog.ShowAsync();
        }
        catch (Exception error) when (error is InvalidOperationException or COMException)
        {
            ShowMessage("暂时无法显示关闭选项，请完成当前对话框后重试：" + error.Message);
            return;
        }
        finally
        {
            _closeDialog = null;
        }
        if (_exiting || result == ContentDialogResult.None) return;
        bool closeToTray = result == ContentDialogResult.Primary;
        if (remember.IsChecked == true)
        {
            SetCloseBehavior(closeToTray, true);
            SaveCloseBehavior();
        }
        if (closeToTray) HideToTray();
        else await ExitApplicationAsync();
    }

    private async Task ExitApplicationAsync()
    {
        if (_exiting) return;
        _exiting = true;
        _closeDialog?.Hide();
        _videoPlayTimer.Stop();
        _videoSeekTimer.Stop();
        _videoPlaying = false;
        UpdateControls();
        PauseButton.IsEnabled = StopButton.IsEnabled = false;
        DebugPauseButton.IsEnabled = DebugStopButton.IsEnabled = false;
        ShowMessage("正在停止任务并退出助手…");
        var workers = _sessions.Values.Select(session => session.Process).Append(_videoWorker).ToArray();
        var results = await Task.WhenAll(workers.Select(StopProcessForExitAsync));
        if (results.Any(success => !success))
        {
            _exiting = false;
            RestoreFromTray();
            UpdateControls();
            ShowMessage("退出未完成，运行组件仍在停止中，请稍后重试。");
            return;
        }
        CloseVideoDebug(updateControls: false);
        _allowWindowClose = true;
        Close();
    }

    private static async Task<bool> StopProcessForExitAsync(Process? process)
    {
        if (process is null) return true;
        try
        {
            if (process.HasExited) return true;
            try
            {
                await process.StandardInput.WriteLineAsync("{\"command\":\"stop\"}");
                await process.StandardInput.FlushAsync();
                process.StandardInput.Close();
            }
            catch (Exception error) when (error is IOException or InvalidOperationException) { }
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(12));
            try { await process.WaitForExitAsync(timeout.Token); }
            catch (OperationCanceledException)
            {
                if (!process.HasExited) process.Kill(true);
                await process.WaitForExitAsync();
            }
            return true;
        }
        catch (InvalidOperationException) { return true; } // Already disposed by the normal exit observer.
        catch (Win32Exception) { return false; }
    }

    private void DisposeTrayIntegration()
    {
        AppWindow.Closing -= AppWindow_Closing;
        AppWindow.Changed -= AppWindow_Changed;
        _trayIcon?.Dispose();
        _trayIcon = null;
    }
}

internal sealed class NativeTrayIcon : IDisposable
{
    private const uint CallbackMessage = 0x8000 + 42;
    private const uint IconId = 1;
    private readonly nint _window;
    private readonly SubclassProc _callback;
    private readonly uint _taskbarCreated;
    private nint _icon;
    private bool _ownsIcon;
    private bool _registered;
    private bool _disposed;

    internal event Action? RestoreRequested;
    internal event Action? ExitRequested;
    internal event Action? Unavailable;

    internal NativeTrayIcon(nint window, string iconPath)
    {
        _window = window;
        _callback = WindowProcedure;
        _icon = LoadImage(0, iconPath, 1, 32, 32, 0x10);
        _ownsIcon = _icon != 0;
        if (_icon == 0) _icon = LoadIcon(0, new nint(32512));
        _taskbarCreated = RegisterWindowMessage("TaskbarCreated");
        if (!SetWindowSubclass(window, _callback, 1, 0))
        {
            if (_ownsIcon) DestroyIcon(_icon);
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        EnsureRegistered();
    }

    internal bool EnsureRegistered()
    {
        if (_disposed) return false;
        if (_registered) return true;
        var data = Data();
        _registered = ShellNotifyIcon(0, ref data);
        if (_registered)
        {
            data.Version = 4;
            if (!ShellNotifyIcon(4, ref data))
            {
                ShellNotifyIcon(2, ref data);
                _registered = false;
            }
        }
        return _registered;
    }

    private NotifyIconData Data() => new()
    {
        Size = (uint)Marshal.SizeOf<NotifyIconData>(), Window = _window, Id = IconId,
        Flags = 1 | 2 | 4 | 0x80, Callback = CallbackMessage, Icon = _icon,
        Tip = "忍3助手", Info = "", InfoTitle = ""
    };

    private nint WindowProcedure(nint window, uint message, nuint wParam, nint lParam, nuint id, nuint data)
    {
        try
        {
            if (_taskbarCreated != 0 && message == _taskbarCreated)
            {
                _registered = false;
                if (!EnsureRegistered()) Unavailable?.Invoke();
            }
            else if (message == CallbackMessage)
            {
                uint notification = (uint)(lParam.ToInt64() & 0xffff);
                if (notification is 0x400 or 0x401 or 0x203) RestoreRequested?.Invoke();
                else if (notification == 0x7b) ShowMenu(wParam);
                return 0;
            }
            else if (message == 0x82) Dispose();
        }
        catch (Exception) { Unavailable?.Invoke(); }
        return DefSubclassProc(window, message, wParam, lParam);
    }

    private void ShowMenu(nuint position)
    {
        nint menu = CreatePopupMenu();
        if (menu == 0) return;
        try
        {
            AppendMenu(menu, 0, 1, "显示主窗口");
            AppendMenu(menu, 0, 2, "退出");
            int x = (short)(position & 0xffff), y = (short)((position >> 16) & 0xffff);
            if (x == -1 && y == -1 && GetCursorPos(out var point)) { x = point.X; y = point.Y; }
            SetForegroundWindow(_window);
            uint selected = TrackPopupMenuEx(menu, 0x100 | 0x80 | 0x20 | 0x2, x, y, _window, 0);
            PostMessage(_window, 0, 0, 0);
            if (selected == 1) RestoreRequested?.Invoke();
            else if (selected == 2) ExitRequested?.Invoke();
        }
        finally { DestroyMenu(menu); }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        var data = Data();
        if (_registered) ShellNotifyIcon(2, ref data);
        RemoveWindowSubclass(_window, _callback, 1);
        if (_ownsIcon) DestroyIcon(_icon);
        _registered = false;
        _icon = 0;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct NotifyIconData
    {
        public uint Size;
        public nint Window;
        public uint Id, Flags, Callback;
        public nint Icon;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string Tip;
        public uint State, StateMask;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)] public string Info;
        public uint Version;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)] public string InfoTitle;
        public uint InfoFlags;
        public Guid Guid;
        public nint BalloonIcon;
    }

    [StructLayout(LayoutKind.Sequential)] private struct Point { public int X, Y; }
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint SubclassProc(nint window, uint message, nuint wParam, nint lParam, nuint id, nuint data);

    [DllImport("shell32.dll", EntryPoint = "Shell_NotifyIconW", CharSet = CharSet.Unicode)]
    [return: MarshalAs(UnmanagedType.Bool)] private static extern bool ShellNotifyIcon(uint message, ref NotifyIconData data);
    [DllImport("comctl32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)] private static extern bool SetWindowSubclass(nint window, SubclassProc callback, nuint id, nuint data);
    [DllImport("comctl32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)] private static extern bool RemoveWindowSubclass(nint window, SubclassProc callback, nuint id);
    [DllImport("comctl32.dll")] private static extern nint DefSubclassProc(nint window, uint message, nuint wParam, nint lParam);
    [DllImport("user32.dll", EntryPoint = "RegisterWindowMessageW", CharSet = CharSet.Unicode)] private static extern uint RegisterWindowMessage(string message);
    [DllImport("user32.dll", EntryPoint = "LoadImageW", CharSet = CharSet.Unicode)] private static extern nint LoadImage(nint instance, string path, uint type, int width, int height, uint flags);
    [DllImport("user32.dll", EntryPoint = "LoadIconW")] private static extern nint LoadIcon(nint instance, nint name);
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool DestroyIcon(nint icon);
    [DllImport("user32.dll")] private static extern nint CreatePopupMenu();
    [DllImport("user32.dll", EntryPoint = "AppendMenuW", CharSet = CharSet.Unicode)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool AppendMenu(nint menu, uint flags, nuint id, string text);
    [DllImport("user32.dll")] private static extern uint TrackPopupMenuEx(nint menu, uint flags, int x, int y, nint window, nint parameters);
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool DestroyMenu(nint menu);
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool GetCursorPos(out Point point);
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool SetForegroundWindow(nint window);
    [DllImport("user32.dll", EntryPoint = "PostMessageW")] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool PostMessage(nint window, uint message, nuint wParam, nint lParam);
}

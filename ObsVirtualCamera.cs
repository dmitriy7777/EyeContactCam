using System.Diagnostics;
using System.IO;
namespace EyeContactCam;
public sealed class ObsVirtualCamera
{
 private static readonly string[] Paths=[@"C:\Program Files\obs-studio\bin\64bit\obs64.exe",@"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe"];
 public string? Executable=>Paths.FirstOrDefault(File.Exists); public bool IsInstalled=>Executable is not null;
 public string Start(){if(Executable is null)return "OBS Studio не найден";Process.Start(new ProcessStartInfo(Executable,"--startvirtualcam --minimize-to-tray --disable-shutdown-check"){WorkingDirectory=Path.GetDirectoryName(Executable)!,UseShellExecute=true});return "OBS Virtual Camera запускается";}
 public void OpenDownloadPage()=>Process.Start(new ProcessStartInfo("https://obsproject.com/download"){UseShellExecute=true});
}

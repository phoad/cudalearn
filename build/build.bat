set PATH=C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\BIN\amd64;C:\Windows\Microsoft.NET\Framework64\v4.0.30319;C:\Windows\Microsoft.NET\Framework64\v3.5;C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\VCPackages;C:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\IDE;C:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\Tools;C:\Program Files (x86)\HTML Help Workshop;C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\bin\NETFX 4.0 Tools\x64;C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\bin\x64;C:\Program Files (x86)\Microsoft SDKs\Windows\v7.0A\bin;C:\Program Files (x86)\AMD APP\bin\x86_64;C:\Program Files (x86)\AMD APP\bin\x86;C:\Users\Phoad\Documents\AMD APP\bin\x86_64;C:\Users\Phoad\Documents\AMD APP\bin\x86;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\bin\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\bin\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.0\libnvvp\;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\libnvvp\;C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.1\\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.1\libnvvp\;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;c:\Program Files (x86)\Microsoft SQL Server\90\Tools\binn\;C:\Program Files (x86)\CMake 2.8\bin;C:\Program Files\TortoiseSVN\bin;C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.1\C\common\bin;C:\Program Files (x86)\NVIDIA Corporation\Cg\bin;C:\Program Files (x86)\NVIDIA Corporation\Cg\bin.x64;C:\Program Files\doxygen\bin;C:\ProgramData\NVIDIA Corporation\NVIDIA GPU Computing SDK 4.2\C\common\bin;C:\Program Files\MATLAB\R2011b\runtime\win64;C:\Program Files\MATLAB\R2011b\bin;C:\Program Files (x86)\Git\cmd;C:\Program Files (x86)\MiKTeX 2.9\miktex\bin\;C:\Program Files (x86)\CMake 2.8\bin;C:\Program Files\TortoiseGit\bin;C:\phoad\tools\apache-maven-3.0.4\bin;C:\Program Files\Java\jdk1.6.0_24\bin;C:\Users\Phoad\Documents\GitHub\ninja 

cd winbuild
cmake -G "Visual Studio 10" ../src
devenv ALL_BUILD.vcxproj /build "release"
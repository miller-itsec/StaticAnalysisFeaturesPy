
rule DllSideloadIndicators
{
    meta:
        description = "Detects string artifacts used in DLL sideloading"
    strings:
        $s1 = "LoadLibrary" ascii
        $s2 = "AppData" ascii
        $s3 = "msiexec.exe" ascii
    condition:
        all of them
}

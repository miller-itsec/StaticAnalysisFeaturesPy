
rule AmsiBypass
{
    meta:
        description = "Common AMSI bypass string (PowerShell malware)"
    strings:
        $a1 = "amsiInitFailed" ascii
        $a2 = "System.Management.Automation" ascii
    condition:
        any of them
}

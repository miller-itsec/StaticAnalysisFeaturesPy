
rule ProcessHollowingIndicators
{
    meta:
        description = "Detects classic process hollowing functions"
    strings:
        $s1 = "CreateProcessA" ascii
        $s2 = "VirtualAllocEx" ascii
        $s3 = "WriteProcessMemory" ascii
        $s4 = "ResumeThread" ascii
    condition:
        3 of ($s*)
}

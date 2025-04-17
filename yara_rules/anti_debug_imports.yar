import "pe"

rule AntiDebugImports
{
    meta:
        description = "Detects common anti-debugging APIs"
    condition:
        pe.imports("kernel32.dll", "IsDebuggerPresent") or pe.imports("ntdll.dll", "NtQueryInformationProcess")
}

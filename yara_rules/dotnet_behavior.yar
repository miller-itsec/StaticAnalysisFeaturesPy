import "pe"

rule DotNetManagedEntry
{
    meta:
        description = "Detects .NET managed PE file"
    condition:
        pe.imports("mscoree.dll")
}

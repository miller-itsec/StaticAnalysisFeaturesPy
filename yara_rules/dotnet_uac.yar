
rule DotNetAutoElevate
{
    meta:
        description = "Detects .NET apps with embedded autoElevate manifest"
    strings:
        $m1 = "requestedExecutionLevel" ascii
        $m2 = "autoElevate" ascii
    condition:
        all of them
}

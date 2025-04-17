
rule EmbeddedPEBinary
{
    meta:
        description = "Detects embedded executables inside non-PE files"
    strings:
        $mz = "MZ"
        $pe = "PE\x00\x00"
    condition:
        for any i in (0..filesize - 2): ($mz at i and $pe at (i+0x3C))
}

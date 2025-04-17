
rule XorEncodedAscii
{
    meta:
        description = "Detects XOR encoded string blobs, common in droppers"
    condition:
        for any i in (0..filesize - 40): (uint8(i)^uint8(i+1) > 0x20 and uint8(i)^uint8(i+2) > 0x20 and uint8(i)^uint8(i+3) > 0x20)
}

import "pe"

rule MaliciousKernelDriver
{
    meta:
        description = "Detects suspicious kernel-mode drivers"

    condition:
        uint16(0) == 0x5A4D and
        uint32(0x3C) + 0x18 < filesize and
        pe.machine == pe.MACHINE_I386 and
        // Check it's *not* a DLL (i.e. doesn't have IMAGE_FILE_DLL set)
        not pe.characteristics & 0x2000 and
        // Optional: Check for INIT or PAGE sections common in drivers
        for any i in (0..pe.number_of_sections - 1): (
            pe.sections[i].name == "INIT" or pe.sections[i].name == "PAGE"
        )
}

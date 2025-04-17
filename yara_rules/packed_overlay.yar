import "pe"

rule PackedWithOverlay
{
    meta:
        description = "Detects binaries with suspicious overlays, typical for packed malware"
    condition:
        uint32(0) == 0x5A4D and
        filesize > 512000 and
        for any i in (0..pe.number_of_sections - 1): (
            pe.sections[i].name == ".text" and
            pe.sections[i].raw_data_size < pe.sections[i].virtual_size
        ) and
        filesize - pe.sections[pe.number_of_sections - 1].raw_data_offset > 2048
}

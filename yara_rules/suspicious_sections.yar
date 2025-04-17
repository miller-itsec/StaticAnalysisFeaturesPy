import "pe"

rule SuspiciousRWXSection
{
    meta:
        description = "Detects PE files with RWX memory permissions in any section"

    condition:
        for any i in (0..pe.number_of_sections - 1) :
            (
                (pe.sections[i].characteristics & pe.SECTION_MEM_READ) != 0 and
                (pe.sections[i].characteristics & pe.SECTION_MEM_WRITE) != 0 and
                (pe.sections[i].characteristics & pe.SECTION_MEM_EXECUTE) != 0
            )
}

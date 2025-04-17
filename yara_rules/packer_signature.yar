import "pe"

rule UPXPacked
{
    meta:
        description = "Detects UPX packer based on section names"
    condition:
        for any i in (0..pe.number_of_sections - 1) :
            (pe.sections[i].name == ".UPX0" or
             pe.sections[i].name == ".UPX1")
}

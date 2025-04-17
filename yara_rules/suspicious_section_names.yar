import "pe"

rule SuspiciousSectionNames
{
    meta:
        description = "Unusual or obfuscated section names"

    condition:
        for any i in (0..pe.number_of_sections - 1) :
            (
                pe.sections[i].name == ".xyz" or
                pe.sections[i].name == ".asdf" or
                pe.sections[i].name == ".xdata" or
                pe.sections[i].name == ".crypt" or
                pe.sections[i].name == ".fuck" or
                pe.sections[i].name == ".junk" or
                pe.sections[i].name == ".bad" or
                pe.sections[i].name == ".enc" or
                pe.sections[i].name == ".pack"
            )
}

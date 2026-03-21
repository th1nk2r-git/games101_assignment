add_rules("mode.debug", "mode.release")
add_requires("opencv")

target("bezier")
    set_kind("binary")
    add_includedirs("include")
    add_files("src/*.cpp")
    set_languages("c++17")
    add_packages("opencv")
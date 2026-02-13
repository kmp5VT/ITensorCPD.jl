file_path = [
    joinpath(@__DIR__,"..","src","algebra","sparse_sign.c"),
    joinpath(@__DIR__,"..","src","algebra","sparsestack.c")
]

lib_path = joinpath(@__DIR__,"..","src","algebra")
lib_name ="libsparse_sign"

os = Sys.KERNEL == :Darwin ? :Darwin :
     Sys.KERNEL == :Linux  ? :Linux :
     Sys.KERNEL == :Windows ? :Windows :
     error("Unsupported OS")


if os==:Linux
    lib_file = joinpath(lib_path,lib_name*".so")
    compile = `gcc -O3 -fPIC -shared -o $lib_file $file_path`
elseif os == :Darwin
    lib_file = joinpath(lib_path,lib_name*".dylib")
    compile = `gcc -O3 -dynamiclib -shared -o $lib_file $file_path`

elseif os== :Windows
    lib_file = joinpath(lib_path,lib_name*".dll")
    compile = `gcc -O3 -shared -o $lib_file $file_path`
else
    error("unsupported OS: $os")
end
run(compile)

{
  pkgs ?
    import <nixpkgs> {
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    },
}:
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.cudaPackages.cuda_nvcc
    pkgs.cudaPackages.cuda_cudart
    pkgs.cudaPackages.cuda_nvprof
    pkgs.cudaPackages.cuda_gdb
   pkgs.cudaPackages_11.cuda_memcheck

    pkgs.gcc13
    pkgs.python312Packages.pandas
    pkgs.python312Packages.matplotlib
    pkgs.valgrind
    pkgs.clang-tools
  ];

  shellHook = ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib

        # Create .clangd file with package paths
        cat > .clangd <<EOF
    CompileFlags:
      Add:
        - --cuda-path=${pkgs.cudatoolkit}
        - --cuda-gpu-arch=sm_50
        - -L${pkgs.cudatoolkit}/lib
        - -I${pkgs.cudatoolkit}/include
        - -I/include
        - -I/src

    EOF
  '';
}

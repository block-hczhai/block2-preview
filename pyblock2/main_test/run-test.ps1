#!/usr/bin/env pwsh

for ($i = 0; $i -le 49; $i++) {
    $formattedI = "{0:D3}" -f $i
    $inputFile = "$formattedI-main.in"
    $outputFile = "$formattedI-main.out"

    Write-Output "TEST $formattedI $(Get-Content -First 1 $inputFile)"

    $depLine = Get-Content $inputFile | Select-String -Pattern "#DEP"
    $depLineProcessed = $depLine -replace "#DEP", "" -replace "^\s+", ""
    $deps = $depLineProcessed -split "\s+"

    foreach ($j in $deps) {
        $formattedJ = "{0:D3}" -f [int]$j
        $depInputFile = "$formattedJ-main.in"
        $depOutputFile = "$formattedJ-main.out"

        Write-Output "-- DEP $formattedJ $(Get-Content -First 1 $depInputFile)"

        python $env:block2main $depInputFile > $depOutputFile
        if ($LASTEXITCODE -ne 0) {
            Get-Content $depOutputFile | Write-Output
            Write-Output "$formattedJ DEP FAILED!"
            exit 3
        }

        python "$formattedJ-check.py" $depOutputFile
        Remove-Item $depOutputFile
    }

    python $env:block2main $inputFile > $outputFile
    if ($LASTEXITCODE -ne 0) {
        Get-Content $outputFile | Write-Output
        Write-Output "$formattedI RUN FAILED!"
        exit 1
    }

    python "$formattedI-check.py" $outputFile
    if ($LASTEXITCODE -ne 0) {
        Get-Content $outputFile | Write-Output
        Write-Output "$formattedI WRONG NUMBER!"
        exit 2
    }

    Remove-Item -Recurse $outputFile, "node0", "nodex"
}

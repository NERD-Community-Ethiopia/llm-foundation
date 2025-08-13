# PowerShell orchestrator to run per-branch pipelines.
# Usage examples:
#   pwsh -File scripts/run_branch_pipelines.ps1 -Pipeline data-prep
#   pwsh -File scripts/run_branch_pipelines.ps1 -Pipeline transformer

param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('data-prep','transformer')]
  [string]$Pipeline,
  [string[]]$Inputs = @('data/raw/*.tsv'),
  [string]$OutDir = 'data/processed/am-om',
  [switch]$SPM,
  [int]$VocabSize = 32000,
  [ValidateSet('unigram','bpe','char','word')]
  [string]$ModelType = 'unigram'
)

$ErrorActionPreference = 'Stop'

function Ensure-Venv {
  if (-not (Test-Path ..\venv\Scripts\python.exe)) {
    throw 'Virtual environment not found at ..\venv\Scripts\python.exe'
  }
}

function Run-DataPrep {
  $env:PYTHONUTF8 = '1'
  $env:PYTHONPATH = (Resolve-Path .\src).Path
  $argsList = @(
    '.\scripts\run_data_preparation_pipeline.py',
    '--inputs'
  ) + $Inputs + @(
    '--out-dir', $OutDir
  )
  if ($SPM) {
    $argsList += @('--spm', '--vocab-size', $VocabSize, '--model-type', $ModelType)
  }
  & ..\venv\Scripts\python.exe @argsList
}

function Run-TransformerPipeline {
  $env:PYTHONUTF8 = '1'
  $env:MPLBACKEND = 'Agg'
  $env:PYTHONPATH = (Resolve-Path .\src).Path
  & ..\venv\Scripts\python.exe .\run_transformer_pipeline.py
}

Ensure-Venv

switch ($Pipeline) {
  'data-prep' { Run-DataPrep }
  'transformer' { Run-TransformerPipeline }
}



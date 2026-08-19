param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [string]$BaseUrl = "http://127.0.0.1:5000",

    [ValidateRange(1, 16)]
    [int]$ChunkSizeMiB = 8,

    [ValidateRange(1, 50)]
    [int]$MaxRetries = 10,

    [ValidateRange(1, 300)]
    [int]$RetrySeconds = 5,

    [string]$StatePath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Net.Http

function Get-StringSha256([string]$Text) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $hash = $sha.ComputeHash($bytes)
        return (-join ($hash | ForEach-Object { $_.ToString("x2") }))
    }
    finally {
        $sha.Dispose()
    }
}

function Get-BytesSha256([byte[]]$Bytes) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $hash = $sha.ComputeHash($Bytes)
        return (-join ($hash | ForEach-Object { $_.ToString("x2") }))
    }
    finally {
        $sha.Dispose()
    }
}

function Read-JsonResponse(
    [System.Net.Http.HttpResponseMessage]$Response,
    [string]$RequestDescription
) {
    $text = $Response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $Response.IsSuccessStatusCode) {
        throw "HTTP $([int]$Response.StatusCode) during ${RequestDescription}: $text"
    }
    try {
        return ($text | ConvertFrom-Json)
    }
    catch {
        throw "Server returned non-JSON data during ${RequestDescription}: $text"
    }
}

function Get-Json([string]$Uri, [string]$Description) {
    $response = $script:HttpClient.GetAsync($Uri).GetAwaiter().GetResult()
    try {
        return Read-JsonResponse $response $Description
    }
    finally {
        $response.Dispose()
    }
}

function Post-Json([string]$Uri, [hashtable]$Payload, [string]$Description) {
    $json = $Payload | ConvertTo-Json -Compress
    $content = [System.Net.Http.StringContent]::new(
        $json,
        [System.Text.Encoding]::UTF8,
        "application/json")
    try {
        $response = $script:HttpClient.PostAsync($Uri, $content).GetAwaiter().GetResult()
        try {
            return Read-JsonResponse $response $Description
        }
        finally {
            $response.Dispose()
        }
    }
    finally {
        $content.Dispose()
    }
}

function Put-ChunkWithRetry(
    [string]$Uri,
    [byte[]]$Bytes,
    [string]$Digest,
    [int]$ChunkIndex
) {
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        $content = New-Object System.Net.Http.ByteArrayContent -ArgumentList (,$Bytes)
        $content.Headers.ContentType = (
            [System.Net.Http.Headers.MediaTypeHeaderValue]::new(
                "application/octet-stream"))
        $content.Headers.ContentLength = $Bytes.Length
        $request = [System.Net.Http.HttpRequestMessage]::new(
            [System.Net.Http.HttpMethod]::Put,
            $Uri)
        $request.Headers.ExpectContinue = $false
        [void]$request.Headers.TryAddWithoutValidation("X-Chunk-Sha256", $Digest)
        $request.Content = $content
        try {
            $response = $script:HttpClient.SendAsync($request).GetAwaiter().GetResult()
            try {
                return Read-JsonResponse $response "uploading chunk $ChunkIndex"
            }
            finally {
                $response.Dispose()
            }
        }
        catch {
            if ($attempt -ge $MaxRetries) {
                throw
            }
            $delay = [Math]::Min(
                60,
                [int]($RetrySeconds * [Math]::Pow(2, $attempt - 1)))
            Write-Warning (
                "Chunk {0} attempt {1}/{2} failed: {3}. Retrying in {4}s." -f
                $ChunkIndex, $attempt, $MaxRetries, $_.Exception.Message, $delay)
            Start-Sleep -Seconds $delay
        }
        finally {
            $request.Dispose()
        }
    }
}

$file = Get-Item -LiteralPath $FilePath -ErrorAction Stop
if ($file.PSIsContainer) {
    throw "FilePath must point to an MP4 file."
}
if ($file.Extension.ToLowerInvariant() -ne ".mp4") {
    throw "Only MP4 files are supported."
}
if ($file.Length -le 0) {
    throw "The MP4 file is empty."
}

$BaseUrl = $BaseUrl.TrimEnd("/")
$chunkBytes = $ChunkSizeMiB * 1MB
$identity = "{0}|{1}|{2}|{3}|{4}" -f @(
    $file.FullName.ToLowerInvariant(),
    $file.Length,
    $file.LastWriteTimeUtc.Ticks,
    $BaseUrl.ToLowerInvariant(),
    $chunkBytes)

if ([string]::IsNullOrWhiteSpace($StatePath)) {
    $stateDirectory = Join-Path $env:LOCALAPPDATA "TennisUploader"
    New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
    $StatePath = Join-Path $stateDirectory ((Get-StringSha256 $identity) + ".json")
}

$script:HttpClient = New-Object System.Net.Http.HttpClient
$script:HttpClient.Timeout = [TimeSpan]::FromMinutes(10)
$state = $null

try {
    $health = Get-Json "$BaseUrl/health" "checking server health"
    if ($health.status -ne "ok") {
        throw "Server health check did not return status=ok."
    }

    if (Test-Path -LiteralPath $StatePath) {
        $candidate = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
        if ($candidate.identity -eq $identity) {
            try {
                $remote = Get-Json (
                    "$BaseUrl/uploads/$($candidate.upload_id)") "resuming upload"
                $state = $candidate
                Write-Host "Resuming upload $($state.upload_id)."
            }
            catch {
                if ($_.Exception.Message -like "HTTP 404*") {
                    Remove-Item -LiteralPath $StatePath -Force
                }
                else {
                    throw
                }
            }
        }
    }

    if ($null -eq $state) {
        $remote = Post-Json "$BaseUrl/uploads/init" @{
            original_filename = $file.Name
            total_size = $file.Length
            chunk_size = $chunkBytes
        } "initializing upload"
        $state = [pscustomobject]@{
            identity = $identity
            upload_id = $remote.job_id
            file_path = $file.FullName
            file_size = $file.Length
            chunk_size = $chunkBytes
            base_url = $BaseUrl
        }
        $state | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8
        Write-Host "Created upload $($state.upload_id)."
    }

    if ($remote.state -notin @("uploading", "assembling")) {
        Write-Host "Upload is already in state '$($remote.state)'."
        Write-Host "Job ID: $($remote.job_id)"
        Write-Host "Status: $BaseUrl/jobs/$($remote.job_id)"
        $remote | ConvertTo-Json -Depth 6
        exit 0
    }

    $received = New-Object "System.Collections.Generic.HashSet[int]"
    foreach ($chunkIndex in @($remote.received_chunks)) {
        [void]$received.Add([int]$chunkIndex)
    }

    $chunkCount = [int]$remote.chunk_count
    $uploadedBytes = [long]$remote.uploaded_bytes
    $stream = [System.IO.File]::Open(
        $file.FullName,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read)
    try {
        for ($chunkIndex = 0; $chunkIndex -lt $chunkCount; $chunkIndex++) {
            if ($received.Contains($chunkIndex)) {
                continue
            }

            $offset = [long]$chunkIndex * [long]$chunkBytes
            $length = [int][Math]::Min([long]$chunkBytes, $file.Length - $offset)
            $buffer = New-Object byte[] $length
            [void]$stream.Seek($offset, [System.IO.SeekOrigin]::Begin)
            $readTotal = 0
            while ($readTotal -lt $length) {
                $read = $stream.Read($buffer, $readTotal, $length - $readTotal)
                if ($read -le 0) {
                    throw "Unexpected end of file while reading chunk $chunkIndex."
                }
                $readTotal += $read
            }

            $digest = Get-BytesSha256 $buffer
            $uri = "$BaseUrl/uploads/$($state.upload_id)/chunks/$chunkIndex"
            $remote = Put-ChunkWithRetry $uri $buffer $digest $chunkIndex
            $uploadedBytes = [long]$remote.uploaded_bytes
            $percent = [Math]::Min(
                100,
                [Math]::Round(100.0 * $uploadedBytes / $file.Length, 2))
            Write-Progress `
                -Activity "Uploading $($file.Name)" `
                -Status "$uploadedBytes / $($file.Length) bytes ($percent%)" `
                -PercentComplete $percent
            Write-Host (
                "Chunk {0}/{1} uploaded; {2}% complete." -f
                ($chunkIndex + 1), $chunkCount, $percent)
        }
    }
    finally {
        $stream.Dispose()
    }

    Write-Progress -Activity "Uploading $($file.Name)" -Completed
    $result = Post-Json "$BaseUrl/uploads/$($state.upload_id)/complete" @{} `
        "completing upload"
    Write-Host "Upload assembled successfully."
    Write-Host "Job ID: $($result.job_id)"
    Write-Host "Status: $BaseUrl/jobs/$($result.job_id)"
    $result | ConvertTo-Json -Depth 6
}
finally {
    $script:HttpClient.Dispose()
}

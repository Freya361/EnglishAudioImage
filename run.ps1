# Set your Gemini API key here if it's not already in your environment
# $env:GEMINI_API_KEY = "AIza..."

if (-not $env:GEMINI_API_KEY) {
    Write-Host "ERROR: GEMINI_API_KEY is not set." -ForegroundColor Red
    Write-Host "Set it with: `$env:GEMINI_API_KEY = 'AIza...'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting English Phrase Card Generator at http://localhost:5000" -ForegroundColor Cyan
python app.py

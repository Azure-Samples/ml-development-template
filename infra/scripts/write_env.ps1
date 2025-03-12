$envFilePath = ".env"

If (Test-Path $envFilePath) {
    Remove-Item $envFilePath -Force
}
New-Item -Path $envFilePath -ItemType File -Force | Out-Null

Add-Content -Path $envFilePath -Value ("AZURE_OPENAI_MODEL_NAME=" + (azd env get-value AZURE_OPENAI_MODEL_NAME))
Add-Content -Path $envFilePath -Value ("AZURE_OPENAI_API_VERSION=" + (azd env get-value AZURE_OPENAI_API_VERSION))
Add-Content -Path $envFilePath -Value ("AZURE_OPENAI_ENDPOINT=" + (azd env get-value AZURE_OPENAI_ENDPOINT))

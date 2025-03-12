envFilePath=".env"

if [ -f "$envFilePath" ]; then
    rm -f "$envFilePath"
fi
touch "$envFilePath"

echo "AZURE_OPENAI_MODEL_NAME=$(azd env get-value AZURE_OPENAI_MODEL_NAME)" >> "$envFilePath"
echo "AZURE_OPENAI_API_VERSION=$(azd env get-value AZURE_OPENAI_API_VERSION)" >> "$envFilePath"
echo "AZURE_OPENAI_ENDPOINT=$(azd env get-value AZURE_OPENAI_ENDPOINT)" >> "$envFilePath"

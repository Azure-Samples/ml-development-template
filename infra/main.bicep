
targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
@allowed(['eastus', 'westus2'])
@metadata({
  azd: {
    type: 'location'
  }
})
param location string = 'eastus'

@description('Name of the resource group')
param resourceGroupName string = ''

@description('Name for the AI resource and used to derive name of dependent resources.')
param aiHubName string = 'hub-demo'

@description('Friendly name for your Hub resource')
param aiHubFriendlyName string = 'Agents Hub resource'

@description('Description of your Azure AI resource displayed in AI studio')
param aiHubDescription string = 'This is an example AI resource for use in Azure AI Studio.'

@description('Name for the AI project resources.')
param aiProjectName string = 'project-demo'

@description('Friendly name for your Azure AI resource')
param aiProjectFriendlyName string = 'Agents Project resource'

@description('Description of your Azure AI resource displayed in AI studio')
param aiProjectDescription string = 'This is an example AI Project resource for use in Azure AI Studio.'

@description('Name for capabilityHost.')
param capabilityHostName string = 'caphost1'

@description('Name of the Azure AI Services account')
param aiServicesName string = 'agentaiservices'

@description('Model name for deployment')
param modelName string = 'gpt-4o'

@description('Model format for deployment')
param modelFormat string = 'OpenAI'

@description('Model version for deployment')
param modelVersion string = '2024-07-18'

@description('Model deployment SKU name')
param modelSkuName string = 'GlobalStandard'

@description('Model deployment capacity')
param modelCapacity int = 50

@description('OpenAI API version')
param openaiApiVersion string = '2024-05-01-preview'

@description('Model deployment location. If you want to deploy an Azure AI resource/model in different location than the rest of the resources created.')
param modelLocation string = 'eastus'

@description('The AI Service Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiServiceAccountResourceId string = ''

@description('The Ai Storage Account full ARM Resource ID. This is an optional field, and if not provided, the resource will be created.')
param aiStorageAccountResourceId string = ''

param timestamp string = utcNow()

// Variables
var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location, timestamp))
var tags = { 'azd-env-name': environmentName }
var name = toLower('${aiHubName}')
var projectName = toLower('${aiProjectName}')

resource resourceGroup 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: '${abbrs.resourcesResourceGroups}${environmentName}' 
  location: location
  tags: tags
}

// Dependent resources for the Azure Machine Learning workspace
module aiDependencies './agent/standard-dependent-resources.bicep' = {
  name: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  scope: resourceGroup
  params: {
    location: location
    storageName: 'st${resourceToken}'
    keyvaultName: 'kv${name}${resourceToken}'
    aiServicesName: '${aiServicesName}${resourceToken}'
    tags: tags

     // Model deployment parameters
     modelName: modelName
     modelFormat: modelFormat
     modelVersion: modelVersion
     modelSkuName: modelSkuName
     modelCapacity: modelCapacity  
     modelLocation: modelLocation

     aiServiceAccountResourceId: aiServiceAccountResourceId
     aiStorageAccountResourceId: aiStorageAccountResourceId
    }
}

module aiHub './agent/standard-ai-hub.bicep' = {
  name: '${abbrs.cognitiveServicesAIhub}${resourceToken}'
  scope: resourceGroup
  params: {
    // workspace organization
    aiHubName: '${name}${resourceToken}'
    aiHubFriendlyName: aiHubFriendlyName
    aiHubDescription: aiHubDescription
    location: location
    tags: tags
    capabilityHostName: '${name}${resourceToken}${capabilityHostName}'

    aiServicesName: aiDependencies.outputs.aiServicesName
    aiServicesId: aiDependencies.outputs.aiservicesID
    aiServicesTarget: aiDependencies.outputs.aiservicesTarget
    
    keyVaultId: aiDependencies.outputs.keyvaultId
    storageAccountId: aiDependencies.outputs.storageId
  }
}


module aiProject './agent/standard-ai-project.bicep' = {
  name: '${abbrs.cognitiveServicesAIproject}${resourceToken}'
  scope: resourceGroup
  params: {
    // workspace organization
    aiProjectName: '${projectName}${resourceToken}'
    aiProjectFriendlyName: aiProjectFriendlyName
    aiProjectDescription: aiProjectDescription
    location: location
    tags: tags
    
    // dependent resources
    capabilityHostName: '${projectName}${resourceToken}${capabilityHostName}'

    aiHubId: aiHub.outputs.aiHubID
    acsConnectionName: aiHub.outputs.acsConnectionName
    aoaiConnectionName: aiHub.outputs.aoaiConnectionName
  }
}

module aiServiceRoleAssignments './agent/ai-service-role-assignments.bicep' = {
  name: 'aiserviceroleassignments${projectName}${resourceToken}deployment'
  scope: resourceGroup
  params: {
    aiServicesName: aiDependencies.outputs.aiServicesName
    aiProjectPrincipalId: aiProject.outputs.aiProjectPrincipalId
    aiProjectId: aiProject.outputs.aiProjectResourceId
  }
}

// App outputs
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output RESOURCE_GROUP string = resourceGroupName
output PROJECT_CONNECTION_STRING string = aiProject.outputs.projectConnectionString
output AZURE_OPENAI_MODEL_NAME string = modelName
output AZURE_OPENAI_API_VERSION string = openaiApiVersion
output AZURE_OPENAI_ENDPOINT string = 'https://${aiServicesName}${resourceToken}.openai.azure.com/'






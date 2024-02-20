import httpx


class EdgeImpulseRestClient:
    def __init__(self, projectApiKey):
        self.base_url = "https://studio.edgeimpulse.com/v1/api/"
        self.headers = {"x-api-key": projectApiKey}

    async def get_project_id(self):
        """Asynchronously retrieves the project ID."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}projects", headers=self.headers
            )
            if response.status_code == 200 and response.json()["success"]:
                return response.json()["projects"][0]["id"]
            else:
                return None

    async def get_deployment_info(self, projectId):
        """Asynchronously retrieves deployment information, including version."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{projectId}/deployment?type=wasm&engine=tflite",
                headers=self.headers,
            )
            if response.status_code == 200 and response.json().get("success"):
                # Returns the version number if available
                version = response.json().get("version")
                return {
                    "version": version,
                    "hasDeployment": response.json().get("hasDeployment"),
                }
            else:
                # Returns None if the request failed or no deployment info was found
                return None

    async def download_model(self, projectId):
        """Asynchronously downloads the model."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{projectId}/deployment/download?type=wasm&engine=tflite",
                headers=self.headers,
            )
            if response.status_code == 200:
                return response.content
            else:
                return None

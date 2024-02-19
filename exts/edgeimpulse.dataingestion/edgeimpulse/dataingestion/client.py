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

    async def check_model_deployment(self, projectId):
        """Asynchronously checks if the model is deployed."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{projectId}/deployment?type=wasm&engine=tflite",
                headers=self.headers,
            )
            if (
                response.status_code == 200
                and response.json().get("success")
                and response.json().get("hasDeployment")
            ):
                return True
            else:
                return False

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

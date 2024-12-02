import httpx

from .impulse import Impulse
from .deployment import DeploymentInfo


class EdgeImpulseRestClient:
    def __init__(self, project_api_key):
        self.base_url = "https://studio.edgeimpulse.com/v1/api/"
        self.headers = {"x-api-key": project_api_key}

    async def get_project_info(self):
        """Asynchronously retrieves the project info."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}projects", headers=self.headers
            )
            if response.status_code == 200 and response.json()["success"]:
                project = response.json()["projects"][0]
                return {"id": project["id"], "name": project["name"]}
            else:
                return None

    async def get_deployment_info(self, project_id):
        """Asynchronously retrieves deployment information, including version."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{project_id}/deployment?type=wasm&engine=tflite",
                headers=self.headers,
            )
            if response.status_code == 200 and response.json().get("success"):
                # Returns the deployment info  if available
                version = response.json().get("version")
                has_deployment = response.json().get("hasDeployment")
                return DeploymentInfo(
                    version=version,
                    has_deployment=has_deployment,
                )
            else:
                # Returns None if the request failed or no deployment info was found
                return None

    async def download_model(self, project_id):
        """Asynchronously downloads the model."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{project_id}/deployment/download?type=wasm&engine=tflite",
                headers=self.headers,
            )
            if response.status_code == 200:
                return response.content
            else:
                return None

    async def get_impulse(self, project_id):
        """Asynchronously fetches the impulse details and returns an Impulse object or None"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{project_id}/impulse",
                headers=self.headers,
            )
            if response.status_code == 200:
                data = response.json()
                if "impulse" in data and data["impulse"].get("inputBlocks"):
                    first_input_block = data["impulse"]["inputBlocks"][0]
                    return Impulse(
                        input_type=first_input_block.get("type"),
                        image_width=first_input_block.get("imageWidth"),
                        image_height=first_input_block.get("imageHeight"),
                    )
                else:
                    return None
            else:
                return None

    async def get_samples_count(self, project_id, category="training"):
        """Asynchronously fetches the number of samples ingested for a specific category"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}{project_id}/raw-data/count?category={category}",
                headers=self.headers,
            )
            if response.status_code == 200:
                data = response.json()
                if "count" in data:
                    return data["count"]
                else:
                    return 0
            else:
                return 0

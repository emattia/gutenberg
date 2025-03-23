from metaflow import CustomFlowDecorator, CustomStepDecorator, kubernetes
from metaflow.user_configs.config_decorators import MutableFlow, MutableStep
import os

NEBUIS_ENDPOINT_URL = "https://storage.eu-north1.nebius.cloud:443"
NEBIUS_BUCKET_PATH = "s3://ob-nebius-test-bucket-1/metaflow-artifacts"


NEBIUS_K8S_CONFIG = dict(
    cpu=100,
    memory=900 * 1000,
    gpu=8,
    shared_memory=200 * 1000,
    image="registry.hub.docker.com/valayob/nebius-nccl-pytorch:0.0.2",
    # image="cr.eu-north1.nebius.cloud/nebius-benchmarks/nccl-tests:2.23.4-ubu22.04-cu12.4",
    node_selector="outerbounds.co/satellite-provider=nebius",  # change this to what ever node selector you have set in your cluster.
    tolerations=[
        {
            "key": "virtual-node.liqo.io/not-allowed",
            "operator": "Equal",
            "value": "true",
            "effect": "NoExecute",
        }
    ],
    # This thing needs a security context of `V1Container` with privilage=true to use Infiniband.
    disk=1000 * 1000,
    use_tmpfs=True,
)

NEBIUS_NODE_SELECTOR = "outerbounds.co/satellite-provider=nebius"
NEBIUS_TOLERATIONS = [
    {
        "key": "virtual-node.liqo.io/not-allowed",
        "operator": "Equal",
        "value": "true",
        "effect": "NoExecute",
    }
]


class nebius_k8s(CustomStepDecorator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init(self, *args, **kwargs):
        self._k8s_args = kwargs
        # add the toleration/node_selector to the k8s_args if not already present
        # if present, then update the existing toleration/node_selector
        k8s_node_selector = self._k8s_args.get("node_selector", None)
        k8s_tolerations = self._k8s_args.get("tolerations", None)
        if k8s_node_selector is None:
            self._k8s_args["node_selector"] = NEBIUS_NODE_SELECTOR
        else:
            if type(k8s_node_selector) == str:
                self._k8s_args["node_selector"] = (
                    k8s_node_selector + "," + NEBIUS_NODE_SELECTOR
                )
            elif type(k8s_node_selector) == dict:
                k8s_node_selector["outerbounds.co/satellite-provider"] = "nebius"
                self._k8s_args["node_selector"] = k8s_node_selector
            else:
                raise ValueError(
                    f"Invalid type for node_selector: {type(k8s_node_selector)}"
                )

        if k8s_tolerations is None:
            self._k8s_args["tolerations"] = NEBIUS_TOLERATIONS
        else:
            # check if nebius toleration is already present
            if any(toleration in k8s_tolerations for toleration in NEBIUS_TOLERATIONS):
                return
            self._k8s_args["tolerations"] = k8s_tolerations + NEBIUS_TOLERATIONS

    def evaluate(self, mutable_step: MutableStep) -> None:
        mutable_step.add_decorator(kubernetes, **self._k8s_args)


class nebius(CustomFlowDecorator):

    def evaluate(self, mutable_flow: MutableFlow) -> None:
        from metaflow import (
            checkpoint,
            model,
            huggingface_hub,
            secrets,
            with_artifact_store,
            project,
        )

        def _add_secrets(step: MutableStep) -> None:
            decos_to_add = []
            swapping_decos = {
                "huggingface_hub": huggingface_hub,
                "model": model,
                "checkpoint": checkpoint,
            }
            already_has_secrets = False
            for d in step.decorators:
                if d.name in swapping_decos:
                    decos_to_add.append((d.name, d.attributes))
                elif d.name == "secrets":
                    already_has_secrets = True
            if already_has_secrets:
                return

            if len(decos_to_add) == 0:
                step.add_decorator(
                    secrets,
                    sources=[
                        "outerbounds.nebuis-bucket-keys",
                        "outerbounds.wandb-keys-so",
                        "outerbounds.hf-keys",
                    ],
                )
                return

            for d, _ in decos_to_add:
                step.remove_decorator(d)

            step.add_decorator(
                secrets,
                sources=[
                    "outerbounds.nebuis-bucket-keys",
                    "outerbounds.wandb-keys-so",
                    "outerbounds.hf-keys",
                ],
            )
            for d, attrs in decos_to_add:
                _deco_to_add = swapping_decos[d]
                step.add_decorator(_deco_to_add, **attrs)

        mutable_flow.add_decorator(
            with_artifact_store,
            type="s3",
            config=lambda: {
                "root": NEBIUS_BUCKET_PATH,
                "client_params": {
                    "aws_access_key_id": os.environ.get("NEBUIS_ACCESS_KEYS"),
                    "aws_secret_access_key": os.environ.get("NEBIUS_SECRET_KEYS"),
                    "endpoint_url": NEBUIS_ENDPOINT_URL,
                },
            },
        )
        mutable_flow.add_decorator(project, name="nebius")
        for step_name, step in mutable_flow.steps:
            _add_secrets(step)

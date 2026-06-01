from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClientRef:
    name: str | None = None
    type: str | None = None
    provider_prefix: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    fallbacks: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: str | dict[str, Any]) -> ClientRef:
        if isinstance(data, str):
            return cls(name=data)
        return cls(
            name=data.get("name"),
            type=data.get("type"),
            provider_prefix=data.get("provider_prefix"),
            model=data.get("model"),
            api_key_env=data.get("api_key_env"),
            fallbacks=data.get("fallbacks", []),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.name is not None:
            result["name"] = self.name
        if self.type is not None:
            result["type"] = self.type
        if self.provider_prefix is not None:
            result["provider_prefix"] = self.provider_prefix
        if self.model is not None:
            result["model"] = self.model
        if self.api_key_env is not None:
            result["api_key_env"] = self.api_key_env
        if self.fallbacks:
            result["fallbacks"] = list(self.fallbacks)
        return result

    @property
    def is_named_ref(self) -> bool:
        return self.name is not None and self.type is None and self.model is None


@dataclass(frozen=True)
class PromptStep:
    name: str
    prompt: str
    client: ClientRef | None = None
    model: str | None = None
    history: list[str] | None = None
    condition: str | None = None
    abort_condition: str | None = None
    system_instructions: str | None = None
    response_format: str | dict | None = None
    response_model: str | None = None
    strict: bool = False
    tools: list[str] | None = None
    tool_choice: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptStep:
        client_raw = data.get("client")
        client = ClientRef.from_dict(client_raw) if client_raw else None

        return cls(
            name=data["name"],
            prompt=data["prompt"],
            client=client,
            model=data.get("model"),
            history=data.get("history"),
            condition=data.get("condition"),
            abort_condition=data.get("abort_condition"),
            system_instructions=data.get("system_instructions"),
            response_format=data.get("response_format"),
            response_model=data.get("response_model"),
            strict=data.get("strict", False),
            tools=data.get("tools"),
            tool_choice=data.get("tool_choice"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "prompt": self.prompt}
        if self.client is not None:
            result["client"] = self.client.to_dict()
        for field_name in (
            "model",
            "history",
            "condition",
            "abort_condition",
            "system_instructions",
            "response_format",
            "response_model",
            "tools",
            "tool_choice",
            "max_tokens",
            "temperature",
        ):
            value = getattr(self, field_name)
            if value is not None and value != []:
                result[field_name] = value
        if self.strict:
            result["strict"] = True
        return result


@dataclass(frozen=True)
class WorkflowDefaults:
    client: ClientRef | None = None
    max_concurrency: int = 10
    strict: bool = False
    system_instructions: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowDefaults:
        client_raw = data.get("client")
        client = ClientRef.from_dict(client_raw) if client_raw else None
        return cls(
            client=client,
            max_concurrency=data.get("max_concurrency", 10),
            strict=data.get("strict", False),
            system_instructions=data.get("system_instructions"),
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
        )


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    description: str = ""
    defaults: WorkflowDefaults = field(default_factory=WorkflowDefaults)
    clients: dict[str, ClientRef] = field(default_factory=dict)
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    prompts: list[PromptStep] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowSpec:
        workflow = data.get("workflow", data)

        defaults_data = workflow.get("defaults", {})
        defaults = WorkflowDefaults.from_dict(defaults_data)

        clients: dict[str, ClientRef] = {}
        for cname, cdata in workflow.get("clients", {}).items():
            clients[cname] = ClientRef.from_dict(cdata)

        tools: dict[str, dict[str, Any]] = {}
        for tname, tdata in workflow.get("tools", {}).items():
            entry = dict(tdata) if isinstance(tdata, dict) else {}
            entry["name"] = tname
            tools[tname] = entry

        prompts: list[PromptStep] = []
        for pdata in workflow.get("prompts", []):
            prompts.append(PromptStep.from_dict(pdata))

        return cls(
            name=workflow.get("name", "unnamed"),
            description=workflow.get("description", ""),
            defaults=defaults,
            clients=clients,
            tools=tools,
            prompts=prompts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": {
                "name": self.name,
                "description": self.description,
                "defaults": {
                    "client": self.defaults.client.to_dict() if self.defaults.client else None,
                    "max_concurrency": self.defaults.max_concurrency,
                    "strict": self.defaults.strict,
                },
                "clients": {n: c.to_dict() for n, c in self.clients.items()},
                "tools": self.tools,
                "prompts": [p.to_dict() for p in self.prompts],
            },
        }

    def get_client_names(self) -> set[str]:
        names: set[str] = set()
        for step in self.prompts:
            if step.client and step.client.name:
                names.add(step.client.name)
        if self.defaults.client and self.defaults.client.name:
            names.add(self.defaults.client.name)
        return names

    def get_tool_names(self) -> set[str]:
        names: set[str] = set()
        for step in self.prompts:
            if step.tools:
                names.update(step.tools)
        return names

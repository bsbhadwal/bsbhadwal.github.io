---
layout: post
title: "Building Self-Discovering AI Agents: Dynamic Tool Discovery with gRPC Reflection and LangGraph"
---

## TL;DR

Traditional AI agents are brittle—they break when APIs change or new services deploy. This post demonstrates a novel architecture that creates truly adaptive agents by combining gRPC Server Reflection with LangGraph. The result: agents that discover, understand, and integrate new tools at runtime without code changes or redeployment.

**Key Benefits:**

- Zero compile-time dependencies between agents and services
- Automatic schema synchronization eliminates integration drift
- Production-ready pattern for microservice-native AI systems
- Seamless handling of service evolution and deployment

## The Problem: Static Agents in Dynamic Environments

Modern distributed systems are built on microservice architectures where service instances are ephemeral, APIs evolve independently, and network topology changes constantly. Yet most AI agents are architected with static tool definitions—a fundamental mismatch that creates operational fragility.

Consider this typical agent deployment scenario:

1. Agent hardcodes tool definitions based on current API specs
2. Microservice team deploys API v2 with new fields
3. Agent breaks or misses new capabilities
4. Manual intervention required to update and redeploy agent

This tight coupling is an anti-pattern that doesn't scale in production environments.

## The Solution: Reflection-Driven Dynamic Discovery

Our approach leverages gRPC Server Reflection—a standardized protocol that allows clients to query service schemas at runtime. While typically used for debugging tools like `grpcurl`, reflection's true power lies in enabling programmatic service discovery.

### Architecture Overview

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│             │    │                  │    │             │
│   LangGraph │────│  gRPC Reflection │────│ Microservice│
│    Agent    │    │   Tool Factory   │    │  Ecosystem  │
│             │    │                  │    │             │
└─────────────┘    └──────────────────┘    └─────────────┘
      ▲                       │
      │                       ▼
      └──── Dynamic Tools ────┘
```

The magic happens in three phases:

1. **Discovery**: Agent queries gRPC servers via reflection protocol
2. **Translation**: Protobuf schemas automatically convert to LLM-compatible JSON Schema
3. **Integration**: Dynamic tools integrate seamlessly into LangGraph workflows

## Implementation Deep Dive

### Phase 1: gRPC Service with Reflection

First, enable reflection on your gRPC services:

```python
# greeter_server.py
import grpc
from grpc_reflection.v1alpha import reflection
import greeter_pb2

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    greeter_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    
    # Enable reflection - this is the key
    SERVICE_NAMES = (
        greeter_pb2.DESCRIPTOR.services_by_name['Greeter'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port('[::]:50051')
    server.start()
```

### Phase 2: Dynamic Tool Factory

The core innovation: a factory that discovers services and manufactures LangChain tools dynamically:

```python
# tool_factory.py
class GrpcToolFactory:
    def __init__(self, host: str, port: int):
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._reflection_db = ProtoReflectionDescriptorDatabase(self._channel)
        self._desc_pool = DescriptorPool(self._reflection_db)
        self._message_factory = MessageFactory(self._desc_pool)

    def discover_tools(self) -> List[Tool]:
        service_names = self._reflection_db.get_services()
        tools = []
        
        for service_name in service_names:
            if service_name == "grpc.reflection.v1alpha.ServerReflection":
                continue  # Skip reflection service itself
                
            service_descriptor = self._desc_pool.FindServiceByName(service_name)
            for method_descriptor in service_descriptor.methods:
                tool = self._create_tool(service_descriptor, method_descriptor)
                tools.append(tool)
        
        return tools

    def _create_tool(self, service_descriptor, method_descriptor) -> Tool:
        # Generate JSON Schema from Protobuf descriptor
        args_schema = self._protobuf_to_json_schema(method_descriptor.input_type)
        
        def _invoke_grpc(**kwargs):
            # Dynamic message creation and invocation
            request_type = self._message_factory.GetPrototype(method_descriptor.input_type)
            request = request_type()
            ParseDict(kwargs, request)
            
            method_path = f"/{service_descriptor.full_name}/{method_descriptor.name}"
            response = self._channel.unary_unary(
                method=method_path,
                request_serializer=request.SerializeToString,
                response_deserializer=self._message_factory.GetPrototype(
                    method_descriptor.output_type
                ).FromString,
            )(request)
            
            return MessageToDict(response, preserving_proto_field_name=True)
        
        return Tool(
            name=f"{service_descriptor.name}_{method_descriptor.name}",
            func=_invoke_grpc,
            description=f"Calls {method_descriptor.name} on {service_descriptor.name}",
            args_schema=args_schema,
        )
```

### Phase 3: LangGraph Integration

The agent remains completely generic—tools are injected at runtime:

```python
# agent.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Dynamic tool discovery at startup
grpc_factory = GrpcToolFactory(host="localhost", port=50051)
tools = grpc_factory.discover_tools()

# Standard LangGraph ReAct pattern
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))  # Dynamic tools injected here

workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")

app = workflow.compile()
```

## The Critical Bridge: Protobuf to JSON Schema Translation

The most complex component is the schema translator. LLMs need JSON Schema for function calling, but gRPC provides Protobuf descriptors. Here's the mapping:

| Protobuf Type | JSON Schema Output |
|---------------|-------------------|
| `string` | `{"type": "string"}` |
| `int32` | `{"type": "integer", "format": "int32"}` |
| `int64` | `{"type": "string", "format": "int64"}` |
| `repeated T` | `{"type": "array", "items": <T schema>}` |
| `message` | `{"type": "object", "properties": {...}}` |

The int64→string mapping is crucial—JSON can't safely represent 64-bit integers without precision loss.

## Production Considerations

### Performance Profile

- **Startup Cost**: Reflection queries add ~100-500ms initialization overhead
- **Runtime Cost**: Zero—standard gRPC performance after discovery
- **Scaling Strategy**: Cache discovered schemas in Redis for rapid agent spawning

### Error Handling

Production agents need robust error handling:

```python
def _invoke_grpc(**kwargs):
    try:
        # ... gRPC call logic
        return response_dict
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            return {"error": "Service temporarily unavailable", "retry": True}
        elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            return {"error": "Request timeout", "retry": True}
        else:
            return {"error": f"gRPC error: {e.details()}", "retry": False}
```

### Schema Evolution

The architecture gracefully handles API evolution:

- **Additive changes**: New optional fields automatically appear in JSON Schema
- **Breaking changes**: Use service versioning (`v1.UserService`, `v2.UserService`)
- **Backward compatibility**: Agents discover and use appropriate service versions

## Advanced Extensions

### Streaming Support

For server-streaming RPCs, the tool wrapper can collect results:

```python
def _invoke_streaming_grpc(**kwargs):
    responses = []
    for response in stream_call(request):
        responses.append(MessageToDict(response))
    return {"stream_results": responses}
```

### Service Registry Integration

Integrate with Consul, etcd, or Kubernetes service discovery:

```python
class ServiceRegistryToolFactory:
    def discover_all_tools(self):
        services = self.registry.discover_grpc_services()
        all_tools = []
        for service in services:
            factory = GrpcToolFactory(service.host, service.port)
            all_tools.extend(factory.discover_tools())
        return all_tools
```

## Real-World Impact

This pattern transforms AI agents from fragile, static components into adaptive participants in microservice ecosystems. Consider these scenarios:

- **DevOps**: Agents automatically discover new monitoring APIs as services deploy
- **Customer Support**: Agents adapt to new CRM integrations without downtime  
- **Data Processing**: Agents leverage newly deployed ML inference services immediately

## Conclusion

By combining gRPC reflection with LangGraph's orchestration capabilities, we've created a blueprint for truly dynamic AI systems. These agents don't just reason—they sense and adapt to their operational environment.

The result is a paradigm shift from building task-specific agents to deploying general-purpose agentic platforms that configure themselves based on available services. This approach eliminates a major source of operational complexity while opening new possibilities for autonomous system integration.

For system architects building the next generation of AI-native platforms, this pattern provides a robust foundation for creating agents that are not just intelligent, but genuinely adaptive to the evolving systems they serve.

---

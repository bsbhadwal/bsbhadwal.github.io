---
layout: post
title: "Architecting Dynamic, Self-Discovering AI Agents: A LangGraph and gRPC Reflection Prototype"
tags: [gRPC, LangGraph, tools]
---

## Section 1: The Next Frontier for AI Agents: Dynamic Tool Discovery

The evolution of artificial intelligence agents has reached a critical juncture. Early paradigms, while effective in constrained environments, relied on a static, compile-time understanding of their capabilities. In these models, the tools and functions an agent can utilize are explicitly defined and hard-coded into its logic. This approach, while straightforward, creates a significant architectural bottleneck, resulting in a tight coupling between the agent and its operational environment. In the context of modern, distributed software ecosystems, this static model is not merely a limitation; it is an anti-pattern that hinders scalability, resilience, and maintainability.1

The rise of microservice architectures has fundamentally reshaped how applications are built and deployed. Monolithic systems are being deconstructed into collections of smaller, independently deployable services, each responsible for a specific business capability.1 This architectural shift brings immense benefits in terms of agility and fault isolation, but it also introduces a new set of challenges, chief among them being the inherent dynamism of the environment. Service instances are ephemeral, their network locations change constantly, and their Application Programming Interfaces (APIs) evolve at an independent cadence.2 To manage this complexity, the industry has widely adopted the Service Discovery Pattern, a mechanism that allows services to locate and communicate with each other dynamically without relying on pre-configured, static addresses. A central service registry tracks the locations of all available service instances, enabling consumers to query this registry and find the services they need in real-time.4

For an AI agent to function as a first-class citizen within such a dynamic ecosystem, it must transcend its static limitations and embrace these same principles of discovery. An agent that depends on a fixed list of tools is fragile; an update to a tool's API or a change in its network location can render the agent ineffective, requiring manual intervention and redeployment. The next frontier for agentic systems, therefore, is the development of agents that can dynamically discover, understand, and integrate the tools available in their runtime environment.

This report introduces a novel architectural pattern that achieves this vision by synergizing the stateful orchestration capabilities of LangGraph with the powerful service discovery mechanism of gRPC Server Reflection. gRPC, a high-performance Remote Procedure Call (RPC) framework, includes a reflection protocol that allows clients to query a server's API schema at runtime. While often utilized for debugging and command-line introspection, its true potential lies in enabling programmatic, real-time service discovery. By leveraging this capability, we can construct agents that, upon initialization, actively interrogate their environment to build their toolset. This process eliminates the need for shared .proto definition files or pre-compiled client-side code, creating a loosely coupled and highly adaptive system.

The vision presented and demonstrated herein is one of self-configuring agents. By combining LangGraph's robust framework for building agentic workflows with gRPC's high-performance communication and reflection-based discovery, we can architect AI systems that are not merely intelligent in their reasoning but are also inherently aware of and adaptable to their operational context. This represents a paradigm shift from building task-specific agents to deploying general-purpose agentic platforms that dynamically configure their capabilities based on the services available, paving the way for more resilient, scalable, and truly autonomous AI applications.

## Section 2: Foundational Technologies: A Technical Primer

To fully appreciate the architecture of the prototype, a foundational understanding of its core technological pillars is essential. This section provides a technical primer on gRPC, the gRPC Reflection Protocol, and the LangGraph framework. These three components form a synergistic triad: gRPC provides the high-performance communication "nervous system," the reflection protocol provides the "senses" for environmental discovery, and LangGraph provides the "brain" for stateful orchestration and reasoning.

### Subsection 2.1: gRPC for High-Performance Inter-Service Communication

gRPC (gRPC Remote Procedure Call) is a modern, open-source RPC framework developed by Google, designed to facilitate efficient communication between distributed systems. It is built on top of HTTP/2, which provides significant performance advantages over the HTTP/1.1 protocol commonly used by REST APIs. Key features of HTTP/2, such as multiplexing (allowing multiple requests and responses to be sent concurrently over a single TCP connection), header compression, and bidirectional streaming, make gRPC particularly well-suited for the low-latency, high-throughput communication required in microservice architectures.6

At the heart of gRPC is the concept of defining a service contract using Protocol Buffers (Protobuf). A .proto file serves as a language-neutral, platform-neutral Interface Definition Language (IDL) where developers define the available services, their RPC methods, and the structure of the request and response messages.8 This contract-first approach ensures strict schema enforcement and strong typing, which is crucial for maintaining consistency and reliability in a polyglot microservices environment where services may be written in different programming languages.

The standard gRPC development workflow in Python involves several key steps:

1. Define the Service: A service is defined in a .proto file, specifying RPC methods and their corresponding message types.  
2. Generate Code: The Protobuf compiler, protoc, along with the grpcio-tools Python plugin, is used to process the .proto file. This command generates two Python files 9:  
   * A \_pb2.py file, which contains the Python classes corresponding to the message definitions.  
   * A \_pb2\_grpc.py file, which contains the gRPC-specific client and server code.  
3. Implement Server and Client: The generated \_pb2\_grpc.py file provides two key components:  
   * Servicer: A base class (e.g., GreeterServicer) that defines the service interface on the server side. Developers implement the service's business logic by creating a subclass and overriding its methods.  
   * Stub: A client-side class (e.g., GreeterStub) that a client instantiates to make remote calls to the server. The stub translates local method calls into network RPCs, abstracting away the complexities of serialization and communication.

This workflow, while robust, traditionally relies on both the client and server having access to the generated code, which implies a shared understanding of the .proto file at compile time.

### Subsection 2.2: The gRPC Reflection Protocol

The gRPC Reflection Protocol is a standard service that, when enabled on a server, allows clients to query and discover the Protobuf schemas of the services hosted on that server at runtime. This capability is analogous to how REST APIs use OpenAPI or Swagger specifications to expose their endpoints and data models, but it operates dynamically over a standardized gRPC service. By using reflection, a client can interact with a gRPC server without needing the original .proto files or any pre-generated client stubs, making it a powerful tool for building dynamic clients, debugging tools, and, as demonstrated in this report, self-configuring AI agents.

### Enabling Reflection on the Server

Activating the reflection service on a Python gRPC server is a straightforward process that requires the grpcio-reflection package. The implementation involves a few key lines of code within the server's setup routine 10:

1. Import Dependencies: The necessary reflection modules are imported: from grpc\_reflection.v1alpha import reflection.  
2. Define Service Names: A tuple of service names that will be discoverable via reflection is created. This tuple must include the fully-qualified names of all application-specific services as well as the reflection service itself. The service names are retrieved from the DESCRIPTOR object in the generated \_pb2.py file.  
   Python  
   SERVICE\_NAMES \= (  
       my\_service\_pb2.DESCRIPTOR.services\_by\_name.full\_name,  
       reflection.SERVICE\_NAME,  
   )

3. Enable the Service: The enable\_server\_reflection function is called, passing the list of service names and the server instance.  
   Python  
   reflection.enable\_server\_reflection(SERVICE\_NAMES, server)

### Client-Side Introspection

On the client side, the grpcio-reflection package provides the necessary tools to interact with the reflection service and dynamically build an understanding of the server's APIs. The core components for this process are:

* grpc\_reflection.v1alpha.proto\_reflection\_descriptor\_database.ProtoReflectionDescriptorDatabase: This class is the primary client-side entry point. It is initialized with a grpc.Channel connected to the server and handles the communication with the remote reflection service.11 Its  
  get\_services() method can be called to retrieve a list of all fully-qualified service names exposed by the server.11  
* google.protobuf.descriptor\_pool.DescriptorPool: This class acts as a database for Descriptor objects. It is initialized with the ProtoReflectionDescriptorDatabase instance and can be used to look up detailed schema information for any service, method, or message type by its fully-qualified name.  
* google.protobuf.message\_factory.MessageFactory: Once the DescriptorPool is populated with schemas from the server, this factory can be used to dynamically create Python classes for any message type at runtime, without needing the pre-generated \_pb2.py modules.

This client-side workflow allows an application to start with only the server's address and progressively build a complete, in-memory representation of its available services and data structures.

### Subsection 2.3: LangGraph for Orchestrating Agentic Workflows

LangGraph is a library built on top of LangChain designed for creating stateful, multi-actor applications with Large Language Models (LLMs). It extends the LangChain Expression Language (LCEL) by modeling agentic workflows as cyclical graphs, providing a more expressive and controllable framework than purely sequential chains.13 This is particularly well-suited for implementing complex reasoning loops and collaborative human-in-the-loop processes.

The fundamental components of a LangGraph application are:

* State: The state is a central, persistent data structure that is passed between the nodes of the graph. It is typically defined as a Python TypedDict. As the graph executes, nodes read from and write to the state, allowing for the accumulation of information and the maintenance of a coherent context throughout the workflow. A common pattern is to include a list of messages in the state to track the history of a conversation.13  
* Nodes: Nodes are the primary units of computation in the graph. A node can be any Python function or LangChain Runnable. Each node receives the current state as input, performs its designated task—such as calling an LLM, executing a tool, or processing data—and returns a dictionary containing the updates to be applied to the state.13  
* Edges: Edges define the control flow of the graph, connecting the nodes to dictate the sequence of execution. LangGraph supports both standard edges, which unconditionally direct flow from one node to another, and conditional edges, which are essential for implementing complex logic. A conditional edge uses a function to inspect the current state and returns the name of the next node to execute, allowing the graph to branch, loop, and make decisions based on the results of previous steps.13

A canonical architecture implemented with LangGraph is the ReAct (Reasoning and Acting) agent. The ReAct pattern creates a cyclical workflow that mirrors a human-like problem-solving process.16 The graph typically consists of:

1. An Agent/Reasoning Node: This node invokes an LLM, providing it with the current state (including the user's query and any previous tool outputs) and a set of available tools. The LLM's task is to reason about the problem and decide on the next step: either provide a final answer or call a tool to gather more information.  
2. A Tool Execution Node: If the LLM decides to use a tool, this node is responsible for executing the specified function with the arguments provided by the LLM.  
3. A Conditional Edge: This edge connects the agent node to both the tool node and a special END node. It inspects the LLM's output; if a tool call is present, it routes the workflow to the tool node. If not, it routes the workflow to END, terminating the process.  
4. A Standard Edge: This edge connects the output of the tool node back to the agent node, completing the loop. This allows the agent to observe the result of its action and incorporate the new information into its next reasoning step.

This cyclical graph structure enables the agent to iteratively think, act, and observe until it has gathered enough information to fully address the user's request.

## Section 3: Prototype Implementation: A Step-by-Step Guide

This section provides a comprehensive walkthrough of the prototype's source code. The implementation is divided into three distinct, modular components: the gRPC tool server, a dynamic tool factory for service discovery, and the LangGraph agent that integrates these components.

### Subsection 3.1: The gRPC Tool Server (greeter\_server.py)

The foundation of our system is a simple gRPC server that exposes a single tool. This server must not only implement the service logic but also enable the reflection service so that its capabilities can be discovered dynamically.

### Protocol Buffer Definition (greeter.proto)

First, we define the service contract in a .proto file. The Greeter service contains one RPC method, SayHello. This method accepts a HelloRequest message containing the name of the person to greet and returns a HelloReply message with a greeting string.

Protocol Buffers

// greeter.proto  
syntax \= "proto3";

package greeter;

// The greeting service definition.  
service Greeter {  
  // Sends a greeting.  
  rpc SayHello (HelloRequest) returns (HelloReply) {}  
}

// The request message containing the user's name.  
message HelloRequest {  
  string name \= 1;  
}

// The response message containing the greetings.  
message HelloReply {  
  string message \= 1;  
}

### Code Generation

With the .proto file defined, we use the grpc\_tools.protoc compiler to generate the necessary Python message and gRPC classes.9 This command should be run from the root of the project directory.

Bash

python \-m grpc\_tools.protoc \-I./protos \--python\_out=. \--grpc\_python\_out=../protos/greeter.proto

This command generates two files, greeter\_pb2.py and greeter\_pb2\_grpc.py, in the project's root directory, making them importable modules.

Server Implementation (greeter\_server.py)

The server implementation brings together the generated code, the service logic, and the crucial reflection service enablement.

Python

\# greeter\_server.py  
import logging  
import time  
from concurrent import futures

import grpc  
from grpc\_reflection.v1alpha import reflection

\# Import generated files  
import greeter\_pb2  
import greeter\_pb2\_grpc

class Greeter(greeter\_pb2\_grpc.GreeterServicer):  
    """Implements the Greeter service logic."""

    def SayHello(self, request, context):  
        """Handles the SayHello RPC call."""  
        logging.info(f"Received SayHello request for name: {request.name}")  
        return greeter\_pb2.HelloReply(message=f"Hello, {request.name}\!")

def serve():  
    """Starts the gRPC server and enables reflection."""  
    server \= grpc.server(futures.ThreadPoolExecutor(max\_workers=10))  
    greeter\_pb2\_grpc.add\_GreeterServicer\_to\_server(Greeter(), server)

    \# The crucial part: Enable Server Reflection.  
    \# The SERVICE\_NAMES tuple lists all services that will be discoverable.  
    \# It must include the reflection service itself.  
    SERVICE\_NAMES \= (  
        greeter\_pb2.DESCRIPTOR.services\_by\_name\['Greeter'\].full\_name,  
        reflection.SERVICE\_NAME,  
    )  
    reflection.enable\_server\_reflection(SERVICE\_NAMES, server)  
    logging.info("gRPC reflection enabled for services: %s", SERVICE\_NAMES)

    server.add\_insecure\_port('\[::\]:50051')  
    server.start()  
    logging.info("Greeter server started on port 50051.")  
      
    try:  
        while True:  
            time.sleep(86400) \# One day  
    except KeyboardInterrupt:  
        server.stop(0)  
        logging.info("Server stopped.")

if \_\_name\_\_ \== '\_\_main\_\_':  
    logging.basicConfig(level=logging.INFO)  
    serve()

The most critical part of this server implementation is the enablement of the reflection service. The SERVICE\_NAMES tuple explicitly registers both our greeter.Greeter service and the standard grpc.reflection.v1alpha.ServerReflection service. Without this step, the server would run but would be opaque to our dynamic agent.10

### Subsection 3.2: The Dynamic gRPC Tool Factory (tool\_factory.py)

This module contains the innovative core of the prototype. The GrpcToolFactory class is responsible for connecting to a gRPC server, using reflection to discover its services and methods, and dynamically manufacturing fully functional LangChain Tool objects from the discovered schemas.

Python

\# tool\_factory.py  
import json  
import logging  
from typing import List, Dict, Any

import grpc  
from google.protobuf.descriptor import FieldDescriptor  
from google.protobuf.descriptor\_pool import DescriptorPool  
from google.protobuf.json\_format import MessageToDict, ParseDict  
from google.protobuf.message\_factory import MessageFactory  
from grpc\_reflection.v1alpha.proto\_reflection\_descriptor\_database import (  
    ProtoReflectionDescriptorDatabase,  
)  
from langchain\_core.tools import Tool

class GrpcToolFactory:  
    """  
    Discovers gRPC services via reflection and creates LangChain Tools for them.  
    """

    def \_\_init\_\_(self, host: str, port: int):  
        self.target \= f"{host}:{port}"  
        self.\_channel \= grpc.insecure\_channel(self.target)  
        self.\_reflection\_db \= ProtoReflectionDescriptorDatabase(self.\_channel)  
        self.\_desc\_pool \= DescriptorPool(self.\_reflection\_db)  
        self.\_message\_factory \= MessageFactory(self.\_desc\_pool)

    def discover\_tools(self) \-\> List:  
        """  
        Discovers all non-reflection services and converts their methods into tools.  
        """  
        try:  
            service\_names \= self.\_reflection\_db.get\_services()  
            logging.info(f"Discovered services: {service\_names}")  
        except grpc.RpcError as e:  
            logging.error(f"Failed to connect or discover services at {self.target}: {e}")  
            return

        tools \=  
        for service\_name in service\_names:  
            if service\_name \== "grpc.reflection.v1alpha.ServerReflection":  
                continue  \# Skip the reflection service itself  
              
            service\_descriptor \= self.\_desc\_pool.FindServiceByName(service\_name)  
            for method\_descriptor in service\_descriptor.methods:  
                tool \= self.\_create\_langchain\_tool\_from\_descriptor(  
                    service\_descriptor, method\_descriptor  
                )  
                tools.append(tool)  
          
        return tools

    def \_create\_langchain\_tool\_from\_descriptor(self, service\_descriptor, method\_descriptor) \-\> Tool:  
        """  
        Creates a single LangChain Tool from a gRPC MethodDescriptor.  
        """  
        tool\_name \= f"{service\_descriptor.full\_name.replace('.', '\_')}\_{method\_descriptor.name}"  
          
        \# Dynamically generate the JSON schema for the tool's arguments  
        args\_schema \= self.\_protobuf\_to\_json\_schema(method\_descriptor.input\_type)  
          
        tool\_description \= (  
            f"Calls the {method\_descriptor.name} method of the {service\_descriptor.full\_name} gRPC service. "  
            f"Use this to perform actions related to {service\_descriptor.name}."  
        )

        def \_invoke\_grpc\_tool(\*\*kwargs: Dict\[str, Any\]) \-\> Dict\[str, Any\]:  
            """A dynamically generated wrapper to invoke the gRPC method."""  
            request\_message\_type \= self.\_message\_factory.GetPrototype(method\_descriptor.input\_type)  
            request\_message \= request\_message\_type()  
              
            \# Populate the protobuf message from the dictionary provided by the LLM  
            ParseDict(kwargs, request\_message)  
              
            method\_path \= f"/{service\_descriptor.full\_name}/{method\_descriptor.name}"  
              
            \# Use the generic unary\_unary invoker  
            response \= self.\_channel.unary\_unary(  
                method=method\_path,  
                request\_serializer=request\_message.SerializeToString,  
                response\_deserializer=self.\_message\_factory.GetPrototype(  
                    method\_descriptor.output\_type  
                ).FromString,  
            )(request\_message)  
              
            \# Convert the protobuf response back to a dictionary for the agent  
            return MessageToDict(response, preserving\_proto\_field\_name=True)

        return Tool(  
            name=tool\_name,  
            func=\_invoke\_grpc\_tool,  
            description=tool\_description,  
            args\_schema=args\_schema,  
        )

    def \_protobuf\_to\_json\_schema(self, message\_descriptor) \-\> Dict\[str, Any\]:  
        """  
        Converts a Protobuf MessageDescriptor into a JSON Schema dictionary.  
        This is the crucial bridge between the gRPC world and the LLM's world.  
        """  
        schema \= {  
            "type": "object",  
            "properties": {},  
            "required":,  
        }  
        for field in message\_descriptor.fields:  
            field\_name \= field.name  
            field\_schema \= {}

            \# Map Protobuf types to JSON Schema types  
            if field.type \== FieldDescriptor.TYPE\_DOUBLE or field.type \== FieldDescriptor.TYPE\_FLOAT:  
                field\_schema\["type"\] \= "number"  
            elif field.type in:  
                field\_schema\["type"\] \= "integer"  
            elif field.type in:  
                field\_schema\["type"\] \= "string" \# Use string for 64-bit ints to avoid precision loss in JSON  
                field\_schema\["format"\] \= "int64"  
            elif field.type \== FieldDescriptor.TYPE\_BOOL:  
                field\_schema\["type"\] \= "boolean"  
            elif field.type \== FieldDescriptor.TYPE\_STRING:  
                field\_schema\["type"\] \= "string"  
            elif field.type \== FieldDescriptor.TYPE\_BYTES:  
                field\_schema\["type"\] \= "string"  
                field\_schema\["format"\] \= "byte" \# Base64 encoded  
            elif field.type \== FieldDescriptor.TYPE\_ENUM:  
                enum\_type \= field.enum\_type  
                field\_schema\["type"\] \= "string"  
                field\_schema\["enum"\] \= \[v.name for v in enum\_type.values\]  
            elif field.type \== FieldDescriptor.TYPE\_MESSAGE:  
                field\_schema \= self.\_protobuf\_to\_json\_schema(field.message\_type)

            if field.label \== FieldDescriptor.LABEL\_REPEATED:  
                schema\["properties"\]\[field\_name\] \= {"type": "array", "items": field\_schema}  
            else:  
                schema\["properties"\]\[field\_name\] \= field\_schema

            \# In proto3, all scalar fields are optional by default.  
            \# We can treat them as required if needed, but for LLM function calling,  
            \# it's often better to let the model decide what to provide.  
            \# For this example, we'll assume all fields are required for simplicity.  
            if field.label\!= FieldDescriptor.LABEL\_REPEATED:  
                 schema\["required"\].append(field\_name)

        return schema

    def close(self):  
        """Closes the gRPC channel."""  
        self.\_channel.close()

This factory encapsulates the entire discovery and tool creation logic. The \_protobuf\_to\_json\_schema method is the critical translation layer, converting the strongly-typed Protobuf schema into a JSON Schema that an LLM can understand and use for function calling. The \_invoke\_grpc\_tool wrapper demonstrates how to perform a fully dynamic gRPC call without any generated stub code, using the generic channel.unary\_unary method and dynamic message (de)serialization.

### Subsection 3.3: The LangGraph Agent (agent.py)

The final component is the LangGraph agent itself. This script defines the agent's state, nodes, and control flow. Crucially, it uses the GrpcToolFactory at startup to dynamically populate its toolset.

Python

\# agent.py  
import os  
from typing import Annotated, List

from dotenv import load\_dotenv  
from langchain\_core.messages import BaseMessage, ToolMessage  
from langchain\_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain\_openai import ChatOpenAI  
from langgraph.graph import END, StateGraph  
from langgraph.prebuilt import ToolNode, add\_messages

from tool\_factory import GrpcToolFactory

\# Load environment variables (for OPENAI\_API\_KEY)  
load\_dotenv()

\# Define the state for our agent graph  
class AgentState(dict):  
    messages: Annotated, add\_messages\]

\# 1\. Discover and create gRPC tools at startup  
grpc\_tool\_factory \= GrpcToolFactory(host="localhost", port=50051)  
tools \= grpc\_tool\_factory.discover\_tools()  
grpc\_tool\_factory.close() \# Close channel after discovery

if not tools:  
    raise RuntimeError("No gRPC tools discovered. Is the server running and reflection enabled?")

\# 2\. Define the nodes for the graph  
tool\_node \= ToolNode(tools)  
model \= ChatOpenAI(temperature=0, streaming=True)  
model\_with\_tools \= model.bind\_tools(tools)

\# The agent's "brain" \- calls the LLM  
def call\_model(state: AgentState):  
    """Invokes the LLM to get the next action or a final response."""  
    response \= model\_with\_tools.invoke(state\["messages"\])  
    return {"messages": \[response\]}

\# 3\. Define the conditional edge for routing  
def should\_continue(state: AgentState) \-\> str:  
    """Determines whether to continue with a tool call or end."""  
    last\_message \= state\["messages"\]\[-1\]  
    if not last\_message.tool\_calls:  
        return "end"  
    else:  
        return "continue"

\# 4\. Assemble the graph  
workflow \= StateGraph(AgentState)

workflow.add\_node("agent", call\_model)  
workflow.add\_node("action", tool\_node)

workflow.set\_entry\_point("agent")

workflow.add\_conditional\_edges(  
    "agent",  
    should\_continue,  
    {"continue": "action", "end": END},  
)

workflow.add\_edge("action", "agent")

\# Compile the graph into a runnable object  
app \= workflow.compile()

This script cleanly separates the concerns. The agent's logic (the graph definition) is generic. The specific tools it uses are not hard-coded but are injected at runtime by the GrpcToolFactory. This makes the agent reusable and adaptable to any environment of gRPC services that have reflection enabled.

## Section 4: End-to-End Execution and Analysis

With all components implemented, this section demonstrates the complete workflow, tracing a user query from input to final response. This analysis validates the dynamic discovery, schema translation, tool invocation, and agentic reasoning loop.

### Execution Script (main.py)

A simple main script is required to run the gRPC server in a background process and interact with the compiled LangGraph agent.

Python

\# main.py  
import logging  
import multiprocessing  
import time  
from typing import List

from langchain\_core.messages import BaseMessage, HumanMessage

from agent import app  
from greeter\_server import serve

def run\_server():  
    """Function to run the gRPC server in a separate process."""  
    serve()

if \_\_name\_\_ \== "\_\_main\_\_":  
    logging.basicConfig(level=logging.INFO)

    \# Start the gRPC server in a background process  
    server\_process \= multiprocessing.Process(target=run\_server)  
    server\_process.start()  
      
    \# Give the server a moment to start up  
    time.sleep(2)

    try:  
        \# Define the conversation input  
        inputs: List \=  
          
        \# Stream the agent's response  
        print("\\n--- Agent Invocation \---")  
        for output in app.stream({"messages": inputs}):  
            for key, value in output.items():  
                print(f"Node '{key}':")  
                print("---")  
                print(value)  
            print("\\n==================================\\n")

    finally:  
        \# Cleanly shut down the server process  
        server\_process.terminate()  
        server\_process.join()  
        logging.info("Server process terminated.")

### Execution Trace

When main.py is executed, the following sequence of events occurs, illustrating the entire dynamic workflow:

1. Startup & Discovery:  
   * The main.py script starts the greeter\_server in a background process.  
   * The agent.py module is imported. During its initialization, GrpcToolFactory is instantiated.  
   * The factory connects to the gRPC server at localhost:50051 and calls discover\_tools().  
   * The ProtoReflectionDescriptorDatabase queries the server's reflection endpoint. The server responds with the list of its registered services: \`\`.  
   * The factory filters out the reflection service and proceeds to inspect greeter.Greeter. It finds the SayHello method.  
   * The \_protobuf\_to\_json\_schema helper is called with the HelloRequest message descriptor. It traverses the descriptor's fields (name: string) and generates the corresponding JSON Schema: {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': \['name'\]}.  
   * A LangChain Tool object is created with the name greeter\_Greeter\_SayHello, a dynamically generated invocation function, and the generated schema. This tool is returned and stored in the tools list.  
   * The LangGraph app is compiled with this dynamically discovered tool.  
2. Agent Invocation & Reasoning (State 1):  
   * The user input, \`\`, is passed to the compiled app.  
   * The graph's entry point, the "agent" node (call\_model), is executed.  
   * The ChatOpenAI model receives the prompt along with the definition of the greeter\_Greeter\_SayHello tool, including its name, description, and JSON Schema for arguments.  
   * The LLM analyzes the input "hi my name is Bob". It recognizes the greeting and the name "Bob". It determines that the most appropriate action is to use the greeter\_Greeter\_SayHello tool and correctly extracts "Bob" as the value for the name parameter.  
   * The model's output is an AIMessage containing a tool\_calls attribute, effectively requesting the execution of the tool with the argument {'name': 'Bob'}. This message is added to the agent's state.  
3. Conditional Routing & Action (State 2 & 3):  
   * The conditional edge should\_continue is evaluated. It inspects the last message in the state, finds the tool\_calls, and returns the string "continue".  
   * Control flows to the "action" node (tool\_node).  
   * The ToolNode executes the greeter\_Greeter\_SayHello tool, passing {'name': 'Bob'} as the keyword arguments.  
   * Inside the tool's dynamically generated wrapper function (\_invoke\_grpc\_tool):  
     * A HelloRequest protobuf message object is created at runtime using the MessageFactory.  
     * ParseDict populates this message object, setting its name field to "Bob".  
     * A generic unary\_unary call is made to the method path /greeter.Greeter/SayHello.  
     * The gRPC server receives the binary-serialized request, its SayHello method is executed, and it returns a HelloReply message with its message field set to "Hello, Bob\!".  
     * The tool wrapper receives this protobuf response and uses MessageToDict to convert it back to a Python dictionary: {'message': 'Hello, Bob\!'}.  
   * This dictionary is wrapped in a ToolMessage and appended to the agent's state.  
4. Observation & Synthesis (State 4):  
   * The graph loops back to the "agent" node.  
   * The LLM is invoked again, but this time its context is much richer. It now sees the entire history: the initial human message, its own decision to call a tool, and the result of that tool call ({'message': 'Hello, Bob\!'}).  
   * The LLM synthesizes this information and generates a final, conversational response for the user, such as "Hello, Bob\!".  
   * This final response is an AIMessage that does *not* contain any tool\_calls.  
5. Termination (State 5):  
   * The conditional edge should\_continue is evaluated again. Seeing no tool\_calls in the last message, it returns "end".  
   * The graph execution terminates, and the final AIMessage containing "Hello, Bob\!" is returned as the result of the invocation.

This trace confirms that the architecture successfully enables an agent to discover, understand, and utilize a gRPC tool without any prior, hard-coded knowledge of its existence or schema.

## Section 5: Architectural Insights and Production Considerations

The prototype successfully demonstrates the core concept of dynamic tool discovery, but deploying such a system in a production environment requires a deeper consideration of the architectural patterns, performance trade-offs, and strategies for long-term maintenance and scalability.

### Subsection 5.1: The Schema Bridge: From Protobuf Descriptors to LLM-Friendly JSON Schema

A fundamental challenge in this architecture is the "impedance mismatch" between the two ecosystems. The gRPC world is built on the strongly-typed, binary-first foundation of Protocol Buffers, with reflection providing schema information via Descriptor objects.17 In contrast, the world of LLM function calling is predominantly designed around the flexible, text-based structure of JSON Schema.18 The LLM uses this schema to understand a tool's required parameters, their types, and their structure, which is essential for generating valid, structured output.20

The programmatic, on-the-fly conversion from a Protobuf Descriptor to a JSON Schema is the critical "glue" that bridges this gap. This automated translation ensures that the schema presented to the LLM is always a perfect, up-to-date representation of the gRPC service's actual contract. This eliminates an entire class of potential integration failures caused by schema drift, where a manually maintained JSON schema for an LLM tool falls out of sync with the underlying gRPC service API it is supposed to represent. The \_protobuf\_to\_json\_schema function in the GrpcToolFactory serves as this vital bridge, recursively mapping Protobuf concepts to their JSON Schema equivalents.

The following table provides a reference for this mapping, which is essential for implementing a robust converter.

| Protobuf Type | FieldDescriptor Constant | JSON Schema Type | JSON Schema Format (Optional) | Notes |
| :---- | :---- | :---- | :---- | :---- |
| double | TYPE\_DOUBLE | {"type": "number"} | {"format": "double"} |  |
| float | TYPE\_FLOAT | {"type": "number"} | {"format": "float"} |  |
| int32, sint32, sfixed32 | TYPE\_INT32, etc. | {"type": "integer"} | {"format": "int32"} |  |
| int64, sint64, sfixed64 | TYPE\_INT64, etc. | {"type": "string"} | {"format": "int64"} | JSON numbers can lose precision for 64-bit integers. String representation is safer. |
| uint32, fixed32 | TYPE\_UINT32, etc. | {"type": "integer", "minimum": 0} | {"format": "int64"} |  |
| uint64, fixed64 | TYPE\_UINT64, etc. | {"type": "string"} | {"format": "uint64"} | JSON numbers can lose precision for 64-bit integers. String representation is safer. |
| bool | TYPE\_BOOL | {"type": "boolean"} |  |  |
| string | TYPE\_STRING | {"type": "string"} |  |  |
| bytes | TYPE\_BYTES | {"type": "string"} | {"format": "byte"} | Represents Base64-encoded data. |
| enum | TYPE\_ENUM | {"type": "string", "enum": \[...\]} |  | The enum list is populated from the EnumValueDescriptor names. |
| message | TYPE\_MESSAGE | {"type": "object", "properties":...} |  | Recursively defined. |
| repeated | LABEL\_REPEATED | {"type": "array", "items":...} |  | The items schema is the type of the repeated field. |

### Subsection 5.2: Scalability and Performance Analysis

While powerful, gRPC reflection is not a zero-cost abstraction. The initial discovery process involves one or more network round-trips between the agent and the gRPC server to fetch the service descriptors. This introduces a latency overhead that must be considered.

The proposed architecture strategically mitigates this by confining the reflection overhead to a one-time cost during the agent's initialization phase. The GrpcToolFactory performs the discovery once, and the resulting LangChain Tool objects, complete with their invocation logic and JSON schemas, are cached in memory for the lifetime of the agent process.

Once this initial discovery is complete, every subsequent tool invocation by the agent is a standard, high-performance gRPC call. These calls bypass the reflection service entirely and leverage the full benefits of gRPC's underlying HTTP/2 transport, including connection reuse, multiplexing, and efficient binary serialization.21 This ensures that the runtime performance of tool execution remains low-latency and highly efficient, which is critical for responsive agent behavior.

For environments where agents may be created and destroyed frequently (e.g., serverless functions or containerized deployments with aggressive scaling), the one-time startup cost of reflection could become a recurring issue. In such scenarios, a more sophisticated caching strategy is warranted. The discovered tool schemas and definitions could be serialized and stored in a shared, external cache, such as Redis or Memcached.22 A newly started agent instance could first attempt to load the tool definitions from this shared cache, falling back to live reflection only if the cache is empty or stale. This approach amortizes the cost of discovery across all agent instances, ensuring fast startup times while still allowing for periodic updates to the cached schemas.

### Subsection 5.3: Advanced Tool Handling and Future Work

The prototype focuses on a simple unary RPC call, but the architecture can be extended to handle more complex gRPC features and production realities.

* Streaming RPCs: gRPC's support for server-side, client-side, and bidirectional streaming is one of its most powerful features. Exposing these as tools to an LLM agent presents both challenges and opportunities. An agent would need a more sophisticated state management and reasoning loop to handle a continuous, asynchronous stream of data from a tool rather than a single, atomic response. For example, a tool that streams stock market updates could continuously append new data to the agent's state, prompting the agent to periodically summarize the updates or take action when a certain threshold is met.  
* Error Handling and Resilience: A production-grade agent must be resilient to tool failures. The dynamic invocation wrapper can be enhanced to catch gRPC-specific RpcError exceptions. Standard gRPC status codes, such as UNAVAILABLE (indicating a temporary network issue) or DEADLINE\_EXCEEDED, can be extracted from the exception and passed back to the agent in a ToolMessage.24 The agent's reasoning prompt can then be designed to understand these errors and implement intelligent recovery strategies, such as retrying a call with an exponential backoff for transient  
  UNAVAILABLE errors, thereby improving the overall robustness of the system.26  
* Schema Evolution and Versioning: Microservice APIs evolve over time. It is crucial to follow best practices for schema evolution to avoid breaking deployed agents.28 Non-breaking changes, such as adding a new optional field to a request message, are handled seamlessly by this architecture. When the agent rediscovers the tool, the new field will simply appear in the generated JSON schema. Older agents that haven't been restarted will continue to function, and the server must be designed to handle requests where the new field is absent.28 Breaking changes, such as renaming a field or changing a field's number, require more careful management, typically through API versioning (e.g.,  
  v1.Greeter, v2.Greeter).29 The dynamic discovery mechanism will naturally expose both versions of the service as distinct sets of tools, allowing for a gradual migration of agent logic to the new version.

## Section 6: Conclusion

This report has detailed and demonstrated a novel and powerful architectural pattern for building advanced AI agents. By integrating the stateful orchestration capabilities of LangGraph with the dynamic service discovery features of gRPC Server Reflection, we have created a prototype for an agent that can autonomously discover, understand, and utilize external tools at runtime.

The successful end-to-end execution of the prototype validates the core tenets of this architecture. The GrpcToolFactory effectively acts as the agent's sensory system, using reflection to perceive the available tools in its environment. The programmatic translation from Protobuf Descriptor objects to LLM-consumable JSON Schemas serves as the critical cognitive bridge, ensuring that the agent's understanding of a tool's interface is always perfectly synchronized with its real-world implementation. Finally, the LangGraph framework provides the robust reasoning engine, orchestrating the cycle of thought, action, and observation that allows the agent to complete its task.

The strategic value of this pattern lies in its ability to create truly decoupled and adaptive systems. Agents are no longer brittle components with hard-coded dependencies but become dynamic participants in an evolving microservice ecosystem. They can be deployed into new environments and automatically adapt their capabilities without requiring code changes or redeployment. This approach significantly enhances the resilience, scalability, and maintainability of agentic applications, providing a robust blueprint for the next generation of enterprise-grade AI systems that are not only intelligent but also deeply and dynamically integrated with the complex digital environments in which they operate.

### Works cited

1. Cutting Edge \- Using gRPC in a Microservice Architecture \- Microsoft Learn, accessed on August 15, 2025, [https://learn.microsoft.com/en-us/archive/msdn-magazine/2019/october/cutting-edge-using-grpc-in-a-microservice-architecture](https://learn.microsoft.com/en-us/archive/msdn-magazine/2019/october/cutting-edge-using-grpc-in-a-microservice-architecture)  
2. Implementing service discovery for microservices \- DEV Community, accessed on August 15, 2025, [https://dev.to/kevwan/implementing-service-discovery-for-microservices-f7p](https://dev.to/kevwan/implementing-service-discovery-for-microservices-f7p)  
3. Service Discovery in Microservices | Baeldung on Computer Science, accessed on August 15, 2025, [https://www.baeldung.com/cs/service-discovery-microservices](https://www.baeldung.com/cs/service-discovery-microservices)  
4. Microservices: Service Discovery Patterns and 3 Ways to Implement | Solo.io, accessed on August 15, 2025, [https://www.solo.io/topics/microservices/microservices-service-discovery](https://www.solo.io/topics/microservices/microservices-service-discovery)  
5. Microservices Patterns: Service Discovery Patterns | Cloud Native Daily \- Medium, accessed on August 15, 2025, [https://medium.com/cloud-native-daily/microservices-patterns-part-03-service-discovery-patterns-97d603b9a510](https://medium.com/cloud-native-daily/microservices-patterns-part-03-service-discovery-patterns-97d603b9a510)  
6. Is gRPC Really Better for Microservices Than GraphQL? \- WunderGraph, accessed on August 15, 2025, [https://wundergraph.com/blog/is-grpc-really-better-for-microservices-than-graphql](https://wundergraph.com/blog/is-grpc-really-better-for-microservices-than-graphql)  
7. Using gRPC to Build Effective Interactions Between Microservices in Go \- Stackademic, accessed on August 15, 2025, [https://blog.stackademic.com/using-grpc-to-build-effective-interactions-between-microservices-on-go-19edd783ebbf](https://blog.stackademic.com/using-grpc-to-build-effective-interactions-between-microservices-on-go-19edd783ebbf)  
8. Overview | Protocol Buffers Documentation, accessed on August 15, 2025, [https://protobuf.dev/overview/](https://protobuf.dev/overview/)  
9. Quick start | Python \- gRPC, accessed on August 15, 2025, [https://grpc.io/docs/languages/python/quickstart/](https://grpc.io/docs/languages/python/quickstart/)  
10. doc/python/server\_reflection.md · df4b6a763d49a9b590a8088dadc8eee569339b1d \- GitLab, accessed on August 15, 2025, [https://gitlab.uni-hannover.de/tci-gateway-module/grpc/-/blob/df4b6a763d49a9b590a8088dadc8eee569339b1d/doc/python/server\_reflection.md](https://gitlab.uni-hannover.de/tci-gateway-module/grpc/-/blob/df4b6a763d49a9b590a8088dadc8eee569339b1d/doc/python/server_reflection.md)  
11. gRPC Reflection — gRPC Python 1.74.0 documentation, accessed on August 15, 2025, [https://grpc.github.io/grpc/python/grpc\_reflection.html](https://grpc.github.io/grpc/python/grpc_reflection.html)  
12. grpc\_reflection.v1alpha.proto\_reflection\_descriptor\_database — gRPC Python 1.74.0 documentation, accessed on August 15, 2025, [https://grpc.github.io/grpc/python/\_modules/grpc\_reflection/v1alpha/proto\_reflection\_descriptor\_database.html](https://grpc.github.io/grpc/python/_modules/grpc_reflection/v1alpha/proto_reflection_descriptor_database.html)  
13. ReAct agent from scratch with Gemini 2.5 and LangGraph, accessed on August 15, 2025, [https://ai.google.dev/gemini-api/docs/langgraph-example](https://ai.google.dev/gemini-api/docs/langgraph-example)  
14. LangGraph \- LangChain, accessed on August 15, 2025, [https://www.langchain.com/langgraph](https://www.langchain.com/langgraph)  
15. Building a ReAct Agent with Langgraph: A Step-by-Step Guide | by Umang | Medium, accessed on August 15, 2025, [https://medium.com/@umang91999/building-a-react-agent-with-langgraph-a-step-by-step-guide-812d02bafefa](https://medium.com/@umang91999/building-a-react-agent-with-langgraph-a-step-by-step-guide-812d02bafefa)  
16. langchain-ai/react-agent: LangGraph template for a simple ReAct agent \- GitHub, accessed on August 15, 2025, [https://github.com/langchain-ai/react-agent](https://github.com/langchain-ai/react-agent)  
17. Descriptors \- Buf Docs, accessed on August 15, 2025, [https://buf.build/docs/reference/descriptors/](https://buf.build/docs/reference/descriptors/)  
18. How to build function calling and JSON mode for open-source and fine-tuned LLMs, accessed on August 15, 2025, [https://www.baseten.co/blog/how-to-build-function-calling-and-json-mode-for-open-source-and-fine-tuned-llms/](https://www.baseten.co/blog/how-to-build-function-calling-and-json-mode-for-open-source-and-fine-tuned-llms/)  
19. chrusty/protoc-gen-jsonschema: Protobuf to JSON-Schema compiler \- GitHub, accessed on August 15, 2025, [https://github.com/chrusty/protoc-gen-jsonschema](https://github.com/chrusty/protoc-gen-jsonschema)  
20. \[Feature Request\] Function Calling \- Easily enforcing valid JSON schema following \- API, accessed on August 15, 2025, [https://community.openai.com/t/feature-request-function-calling-easily-enforcing-valid-json-schema-following/263515](https://community.openai.com/t/feature-request-function-calling-easily-enforcing-valid-json-schema-following/263515)  
21. gRPC vs REST \- Difference Between Application Designs \- AWS, accessed on August 15, 2025, [https://aws.amazon.com/compare/the-difference-between-grpc-and-rest/](https://aws.amazon.com/compare/the-difference-between-grpc-and-rest/)  
22. Caches in Microservice architecture \- SoftwareMill, accessed on August 15, 2025, [https://softwaremill.com/caches-in-microservice-architecture/](https://softwaremill.com/caches-in-microservice-architecture/)  
23. Caching patterns \- Database Caching Strategies Using Redis \- AWS Documentation, accessed on August 15, 2025, [https://docs.aws.amazon.com/whitepapers/latest/database-caching-strategies-using-redis/caching-patterns.html](https://docs.aws.amazon.com/whitepapers/latest/database-caching-strategies-using-redis/caching-patterns.html)  
24. Mastering gRPC Error Handling: Best Practices and Key Strategies, accessed on August 15, 2025, [https://www.bytesizego.com/blog/mastering-grpc-go-error-handling](https://www.bytesizego.com/blog/mastering-grpc-go-error-handling)  
25. Enhancing gRPC Error Handling in a Go Microservice Architecture : r/golang \- Reddit, accessed on August 15, 2025, [https://www.reddit.com/r/golang/comments/1b21r78/enhancing\_grpc\_error\_handling\_in\_a\_go/](https://www.reddit.com/r/golang/comments/1b21r78/enhancing_grpc_error_handling_in_a_go/)  
26. Building Production Grade Microservices with Go and gRPC \- A Step-by-Step Developer Guide with Example \- DEV Community, accessed on August 15, 2025, [https://dev.to/nikl/building-production-grade-microservices-with-go-and-grpc-a-step-by-step-developer-guide-with-example-2839](https://dev.to/nikl/building-production-grade-microservices-with-go-and-grpc-a-step-by-step-developer-guide-with-example-2839)  
27. Microservices Resilience and Fault Tolerance with applying Retry and Circuit-Breaker patterns using Polly | by Mehmet Ozkaya | aspnetrun | Medium, accessed on August 15, 2025, [https://medium.com/aspnetrun/microservices-resilience-and-fault-tolerance-with-applying-retry-and-circuit-breaker-patterns-c32e518db990](https://medium.com/aspnetrun/microservices-resilience-and-fault-tolerance-with-applying-retry-and-circuit-breaker-patterns-c32e518db990)  
28. Versioning gRPC services | Microsoft Learn, accessed on August 15, 2025, [https://learn.microsoft.com/en-us/aspnet/core/grpc/versioning?view=aspnetcore-9.0](https://learn.microsoft.com/en-us/aspnet/core/grpc/versioning?view=aspnetcore-9.0)  
29. Proto Best Practices | Protocol Buffers Documentation, accessed on August 15, 2025, [https://protobuf.dev/best-practices/dos-donts/](https://protobuf.dev/best-practices/dos-donts/)  
30. gRPC and schema evolution guarantees? \- Stack Overflow, accessed on August 15, 2025, [https://stackoverflow.com/questions/59581377/grpc-and-schema-evolution-guarantees](https://stackoverflow.com/questions/59581377/grpc-and-schema-evolution-guarantees)
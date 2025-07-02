#!/usr/bin/env python3
"""
RAG query interface for the viral content documentation with full Together.ai support.
"""

import os
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Qdrant
from together import Together
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.base import BaseLanguageModel
from qdrant_client import QdrantClient
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import click
import cohere
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
console = Console()

from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field
from typing import Any, List, Optional

class TogetherLLM(BaseLLM):
    """LangChain-compatible wrapper for Together.ai models."""
    
    model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    temperature: float = 0.2
    max_tokens: int = 12288
    top_p: float = 0.8
    top_k: int = 40
    client: Any = Field(default=None, exclude=True)
    
    def __init__(self, 
                 model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo", 
                 temperature: float = 0.2,
                 max_tokens: int = 12288,
                 top_p: float = 0.8,
                 top_k: int = 40,
                 **kwargs):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    @property
    def _llm_type(self) -> str:
        return "together"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Together.ai API."""
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=stop
        )
        
        return response.choices[0].message.content
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts."""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)
    
    def invoke(self, input, config=None, **kwargs):
        """Invoke method for LCEL compatibility."""
        if hasattr(input, 'to_messages'):
            formatted_messages = input.to_messages()
            
            together_messages = []
            for msg in formatted_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
                    together_messages.append({"role": role, "content": msg.content})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=together_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k
            )
            
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": str(input)}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k
            )
        
        return response.choices[0].message.content

class RAGQueryInterface:
    def __init__(self):
        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            self.client = QdrantClient(url=qdrant_url)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        # Initialize LLM (Together.ai, Claude, or OpenAI)
        model_name = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        
        if os.getenv("TOGETHER_API_KEY") and model_name.startswith("Qwen"):
            self.llm = TogetherLLM(
                model=model_name,
                temperature=0.2,
                max_tokens=12288,
                top_p=0.8,
                top_k=40,
            )
            console.print(f"[green]Using Together.ai model: {model_name}[/green]")
        elif os.getenv("ANTHROPIC_API_KEY") and model_name.startswith("claude"):
            self.llm = ChatAnthropic(
                model=model_name,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3,
                max_tokens=4096
            )
            console.print(f"[green]Using Claude model: {model_name}[/green]")
        else:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.3
            )
            console.print(f"[green]Using OpenAI model: {model_name}[/green]")
        
        # Setup vector stores for all collections
        self.regular_vectorstore = Qdrant(
            client=self.client,
            collection_name="viral_content_framework",
            embeddings=self.embeddings
        )
        
        self.logic_vectorstore = Qdrant(
            client=self.client,
            collection_name="viral_content_logic_codex",
            embeddings=self.embeddings
        )
        
        self.linkedin_vectorstore = Qdrant(
            client=self.client,
            collection_name="linkedin_optimization",
            embeddings=self.embeddings
        )
        
        # Create prompts and chains
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup the retrieval chains."""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Logic chain with brutal tactical prompt
        logic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a brutal, unforgiving tactical advisor for viral short-form content creation.

CRITICAL INSTRUCTIONS:
- NO reasoning, thinking, or analysis tags
- NO soft encouraging language
- Jump DIRECTLY to harsh tactical rules

RESPONSE STRUCTURE:
1. LEAD with specific tactical rules, formulas, and rigid criteria
2. SUPPORT with context only to fill gaps
3. End with unforgiving tactical summary

TACTICAL PRIORITY:
- Exact algorithmic rules and numerical thresholds
- Cold mathematical formulas and calculations  
- Binary decision criteria (pass/fail triggers)
- Step-by-step tactical procedures
- Measurable benchmarks with no wiggle room

TONE: Direct, harsh, rule-based. No encouragement. Pure tactical guidance.

Context:
{context}

Deliver tactical rules immediately. Be brutal and specific."""),
            ("human", "{question}")
        ])
        
        self.logic_chain = (
            {
                "context": self.logic_vectorstore.as_retriever(search_kwargs={"k": 5}) | format_docs,
                "question": RunnablePassthrough(),
            }
            | logic_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Regular chain
        regular_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant for viral short-form content creation.

Provide direct, actionable answers immediately. Focus on:
- Creating viral hooks and engaging openings
- Understanding social media algorithms
- Short-form video content strategies
- Viral content psychology and engagement tactics

Context:
{context}

Provide detailed, actionable answers with specific examples."""),
            ("human", "{question}")
        ])
        
        self.regular_chain = (
            {
                "context": self.regular_vectorstore.as_retriever(search_kwargs={"k": 5}) | format_docs,
                "question": RunnablePassthrough(),
            }
            | regular_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _detect_query_type(self, question: str, avoid_linkedin: bool = False) -> str:
        """Detect query type with LinkedIn avoidance."""
        question_lower = question.lower()
        
        # Check for LinkedIn avoidance flags first
        linkedin_avoidance_flags = [
            "avoid linkedin", "no linkedin", "skip linkedin", "ignore linkedin",
            "avoid linked in", "no linked in", "skip linked in", "ignore linked in",
            "avoid linkedin advice", "no linkedin advice", "viral only", "tiktok only"
        ]
        
        if avoid_linkedin or any(flag in question_lower for flag in linkedin_avoidance_flags):
            return "logic"
        
        # Check for explicit LinkedIn mentions
        linkedin_keywords = ["linkedin", "linked-in", "linked in"]
        if any(keyword in question_lower for keyword in linkedin_keywords):
            return "linkedin"
        
        # Default to logic codex
        return "logic"
    
    def query(self, question: str, collection: str = "auto", avoid_linkedin: bool = False) -> Dict:
        """Query the RAG system with full parameter support."""
        try:
            # Determine which collection/chain to use
            if collection == "auto":
                query_type = self._detect_query_type(question, avoid_linkedin)
            elif collection in ["logic", "tactical"]:
                query_type = "logic"
            elif collection == "linkedin" and not avoid_linkedin:
                query_type = "linkedin"
            elif collection == "linkedin" and avoid_linkedin:
                query_type = "logic"
                console.print("[yellow]⚠️ LinkedIn collection requested but avoided - using logic codex[/yellow]")
            else:
                query_type = "general"
            
            # Use appropriate chain
            if query_type == "logic":
                console.print("[blue]🎯 Using tactical logic codex[/blue]")
                answer = self.logic_chain.invoke(question)
                source_type = "Logic Codex"
                source_docs = self.logic_vectorstore.similarity_search(question, k=5)
            else:
                console.print("[blue]🎬 Using viral content framework[/blue]")
                answer = self.regular_chain.invoke(question)
                source_type = "Viral Content Framework"
                source_docs = self.regular_vectorstore.similarity_search(question, k=5)
            
            return {
                "answer": answer,
                "source_type": source_type,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in source_docs
                ]
            }
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            return {"answer": "An error occurred while processing your query.", "sources": [], "source_type": "Error"}
    
    def display_result(self, result: Dict):
        """Display the query result nicely."""
        source_type = result.get("source_type", "Viral Content Framework")
        if source_type == "Logic Codex":
            title_color = "yellow"
            icon = "🎯"
        else:
            title_color = "cyan"
            icon = "🎬"
        
        console.print(Panel(
            Markdown(result["answer"]),
            title=f"[bold {title_color}]{icon} Answer ({source_type})[/bold {title_color}]",
            border_style=title_color
        ))
        
        # Display sources
        if result["sources"]:
            console.print(f"\n[bold yellow]Sources ({source_type}):[/bold yellow]")
            for i, source in enumerate(result["sources"], 1):
                metadata = source['metadata']
                console.print(f"\n[dim]{i}. {metadata['source']}[/dim]")
                console.print(f"   Preview: {source['content']}")

@click.command()
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.argument('question', required=False)
def main(interactive: bool, question: str):
    """Query the viral content documentation."""
    console.print("[bold cyan]Viral Content RAG Query Interface[/bold cyan]\n")
    
    # Initialize interface
    interface = RAGQueryInterface()
    
    if interactive or not question:
        # Interactive mode
        console.print("Enter your questions (type 'exit' to quit):\n")
        
        while True:
            question = console.input("[bold green]Question:[/bold green] ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            if not question.strip():
                continue
            
            # Process query
            with console.status("[blue]Searching documentation...[/blue]"):
                result = interface.query(question)
            
            # Display result
            interface.display_result(result)
            console.print("\n" + "="*80 + "\n")
    
    else:
        # Single query mode
        with console.status("[blue]Searching documentation...[/blue]"):
            result = interface.query(question)
        
        interface.display_result(result)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example script demonstrating the Deep Researcher Agent

This script shows how to use the Deep Researcher Agent programmatically
to analyze PDF documents and extract/analyze web content.
"""

import os
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from deep_researcher_agent import DeepResearcherAgent
except ImportError as e:
    print(f"Error importing Deep Researcher Agent: {e}")
    print("Please install the required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def main():
    print("🧠 Deep Researcher Agent - Example Usage")
    print("=" * 50)
    
    # Example PDF path (you should replace this with your own PDF)
    pdf_path = "sample_research_paper.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("\nPlease provide a PDF file to analyze.")
        print("You can:")
        print("1. Place a PDF file named 'sample_research_paper.pdf' in this directory")
        print("2. Or modify the pdf_path variable in this script")
        return
    
    print(f"📄 Analyzing PDF: {pdf_path}")
    
    # Initialize the Deep Researcher Agent
    print("\n🔧 Initializing Deep Researcher Agent...")
    agent = DeepResearcherAgent(
        use_selenium=False,  # Set to True for JavaScript-heavy sites
        delay_between_requests=1.0,  # Be respectful to servers
        max_workers=3  # Conservative number of concurrent workers
    )
    
    # Configuration for this example
    config = {
        'max_links': 10,  # Process up to 10 links
        'allowed_domains': ['arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov'],  # Focus on academic sources
        'blocked_domains': ['facebook.com', 'twitter.com', 'linkedin.com'],  # Skip social media
        'include_images': True,
        'max_depth': 2,  # Go 2 levels deep
        'max_links_per_level': 5,
        'use_multi_level': True
    }
    
    print(f"\n🔍 Configuration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    # Start the research
    print(f"\n🚀 Starting deep research analysis...")
    start_time = time.time()
    
    try:
        result = agent.deep_research(
            pdf_path=pdf_path,
            **config
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Research completed in {duration:.2f} seconds!")
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 RESEARCH RESULTS")
        print("=" * 50)
        
        print(f"📎 Total Links Found: {result.total_links_found}")
        print(f"✅ Successful Scrapes: {result.successful_scrapes}")
        print(f"❌ Failed Scrapes: {result.failed_scrapes}")
        print(f"🔍 Maximum Depth Reached: {result.max_depth_reached}")
        print(f"🌐 Unique Domains: {len(result.domain_analysis)}")
        print(f"🕸️  Network Nodes: {result.content_network.number_of_nodes()}")
        print(f"🔗 Network Edges: {result.content_network.number_of_edges()}")
        
        # Success rate
        if result.total_links_found > 0:
            success_rate = (result.successful_scrapes / result.total_links_found) * 100
            print(f"📈 Overall Success Rate: {success_rate:.1f}%")
        
        # Domain analysis
        if result.domain_analysis:
            print(f"\n🌐 Top Domains:")
            for i, (domain, count) in enumerate(list(result.domain_analysis.items())[:5], 1):
                print(f"  {i}. {domain}: {count} pages")
        
        # Key insights
        if result.key_insights:
            print(f"\n💡 Key Insights:")
            for insight in result.key_insights:
                print(f"  • {insight}")
        
        # Sample content details
        successful_content = [c for c in result.scraped_content if c.success]
        if successful_content:
            print(f"\n📑 Sample Content Details:")
            for i, content in enumerate(successful_content[:3], 1):  # Show first 3
                print(f"  {i}. {content.title[:60]}{'...' if len(content.title) > 60 else ''}")
                print(f"     URL: {content.url}")
                print(f"     Content Length: {len(content.content):,} characters")
                print(f"     Images: {len(content.images)}")
                print(f"     Depth: {content.depth}")
                print()
        
        # Export options
        print("💾 Export Options:")
        print("  • Summary: result.summary")
        print("  • Network: result.content_network (NetworkX DiGraph)")
        print("  • Raw Data: result.scraped_content")
        
        # Example: Save summary to file
        summary_file = f"research_summary_{int(time.time())}.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(result.summary)
        print(f"  • Summary saved to: {summary_file}")
        
        # Example: Basic network analysis
        if result.content_network.number_of_nodes() > 0:
            print(f"\n🕸️  Network Analysis:")
            import networkx as nx
            
            # Basic metrics
            if result.content_network.number_of_nodes() > 1:
                density = nx.density(result.content_network)
                print(f"  • Network density: {density:.3f}")
                
                if nx.is_weakly_connected(result.content_network):
                    print("  • Network is connected")
                else:
                    components = list(nx.weakly_connected_components(result.content_network))
                    print(f"  • Network has {len(components)} components")
            
            # Most connected nodes
            degree_dict = dict(result.content_network.degree())
            if degree_dict:
                most_connected = max(degree_dict.items(), key=lambda x: x[1])
                print(f"  • Most connected page: {most_connected[0][:50]}... ({most_connected[1]} connections)")
        
        print(f"\n🎉 Analysis complete! Check the generated files for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # If it's a common error, provide helpful suggestions
        if "PDF" in str(e):
            print("\n💡 Suggestions:")
            print("  • Check if the PDF file exists and is readable")
            print("  • Try a different PDF file")
            print("  • Ensure the PDF contains actual links/URLs")
        
        elif "selenium" in str(e).lower() or "chromedriver" in str(e).lower():
            print("\n💡 Suggestions:")
            print("  • Set use_selenium=False in the agent configuration")
            print("  • Or install ChromeDriver for Selenium support")
        
        elif "connection" in str(e).lower() or "timeout" in str(e).lower():
            print("\n💡 Suggestions:")
            print("  • Check your internet connection")
            print("  • Increase delay_between_requests")
            print("  • Reduce max_workers for slower connections")

def create_sample_pdf_info():
    """Provide information about creating a sample PDF for testing"""
    print("\n📝 To test the Deep Researcher Agent, you need a PDF with links.")
    print("You can:")
    print("1. Use any research paper from arXiv.org")
    print("2. Use a PDF with web references")
    print("3. Create a simple test PDF with URLs")
    print("\nExample academic PDFs with many references:")
    print("• https://arxiv.org/pdf/1706.03762.pdf (Attention Is All You Need)")
    print("• https://arxiv.org/pdf/2010.11929.pdf (An Image is Worth 16x16 Words)")
    print("• Any paper from Google Scholar, IEEE, or academic conferences")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        create_sample_pdf_info()
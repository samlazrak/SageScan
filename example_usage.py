#!/usr/bin/env python3
"""
Example usage of the OCR Text Summarizer

This script demonstrates how to use the OCR Summarizer classes
programmatically rather than through the command line interface.
"""

from ocr_summarizer import OCRSummarizerApp

def example_text_summarization():
    """Example of summarizing text directly."""
    print("=== Text Summarization Example ===")
    
    # Sample text based on the "think tool" concept
    sample_text = """
    Using the think tool

    Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
    - List the specific rules that apply to the current request
    - Check if all required information is collected
    - Verify that the planned action complies with all policies
    - Iterate over tool results for correctness 

    Here are some examples of what to iterate over inside the think tool:
    
    User wants to cancel flight ABC123
    - Need to verify: user ID, reservation ID, reason
    - Check cancellation rules:
      * Is it within 24h of booking?
      * If not, check ticket class and insurance
    - Verify no segments flown or are in the past
    - Plan: collect missing info, verify rules, get confirmation

    User wants to book 3 tickets to NYC with 2 checked bags each
    - Need user ID to check:
      * Membership tier for baggage allowance
      * Which payments methods exist in profile
    - Baggage calculation:
      * Economy class × 3 passengers
      * If regular member: 1 free bag each → 3 extra bags = $150
      * If silver member: 2 free bags each → 0 extra bags = $0
      * If gold member: 3 free bags each → 0 extra bags = $0
    - Payment rules to verify:
      * Max 1 travel certificate, 1 credit card, 3 gift cards
      * All payment methods must be in profile
      * Travel certificate remainder goes to waste
    - Plan:
    1. Get user ID
    2. Verify membership level for bag fees
    3. Check which payment methods in profile and if their combination is allowed
    4. Calculate total: ticket price + any bag fees
    5. Get explicit confirmation for booking
    """
    
    # Initialize the application
    app = OCRSummarizerApp()
    
    # Process the text
    result = app.process_text(sample_text, output_file="text_summary_example.txt")
    
    # Display results
    print(f"Status: {result['status']}")
    print(f"Original text length: {len(result['extracted_text'])} characters")
    print(f"Summary:\n{result['summary']}")
    print("\nFull results saved to: text_summary_example.txt")


def example_image_processing():
    """Example of processing an image (if available)."""
    print("\n=== Image Processing Example ===")
    
    # This would work if you have an image file
    image_path = "sample_document.jpg"  # Replace with actual image path
    
    try:
        app = OCRSummarizerApp()
        result = app.process_image(image_path, output_file="image_summary_example.txt")
        
        print(f"Status: {result['status']}")
        print(f"Extracted text length: {len(result['extracted_text'])} characters")
        print(f"Summary:\n{result['summary']}")
        print("\nFull results saved to: image_summary_example.txt")
        
    except FileNotFoundError:
        print(f"Image file '{image_path}' not found.")
        print("To test image processing, place an image file in the current directory")
        print("and update the 'image_path' variable in this script.")


def example_with_different_model():
    """Example using a different MLX model."""
    print("\n=== Different Model Example ===")
    
    # Use a smaller, faster model
    app = OCRSummarizerApp(model_name="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    
    text = """
    The think tool concept involves using a structured approach to problem-solving.
    It requires careful analysis of requirements, verification of constraints,
    and systematic planning before taking action. This approach helps ensure
    accuracy and compliance with established policies and procedures.
    """
    
    result = app.process_text(text)
    
    print(f"Model used: {app.summarizer.model_name}")
    print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    print("OCR Text Summarizer - Example Usage\n")
    
    try:
        # Run text summarization example
        example_text_summarization()
        
        # Run image processing example (if image available)
        example_image_processing()
        
        # Run different model example
        example_with_different_model()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  ./setup.sh")
        print("or")
        print("  pip3 install -r requirements.txt")
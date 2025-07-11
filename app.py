import base64
import mimetypes
import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import base64
from PIL import Image
import io
from google.genai import types

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# In-memory storage for the most recent generated image
recent_image_data = {
    'base64': None,
    'mime_type': None,
    'timestamp': None,
    'image_id': None
}

def process_image_data(data, mime_type):
    """Process binary image data and return base64 string"""
    try:
        print(f"Processing image data - Type: {type(data)}, MIME: {mime_type}")
        
        # Handle different data types
        if isinstance(data, str):
            # If data is already a base64 string, decode it
            try:
                image_data = base64.b64decode(data)
                print("Data was base64 string, decoded successfully")
            except Exception as e:
                print(f"Failed to decode base64 string: {e}")
                return None
        elif isinstance(data, bytes):
            # If data is already bytes, use directly
            image_data = data
            print("Data was bytes, using directly")
        else:
            # Try to convert to bytes
            try:
                image_data = bytes(data)
                print("Converted data to bytes")
            except Exception as e:
                print(f"Failed to convert data to bytes: {e}")
                return None
        
        print(f"Image data length: {len(image_data)} bytes")
        
        # Validate that we have actual image data
        if len(image_data) == 0:
            print("Image data is empty")
            return None
        
        # Check for common image headers to validate format
        if image_data[:4] == b'\x89PNG':
            print("Detected PNG format")
        elif image_data[:2] == b'\xff\xd8':
            print("Detected JPEG format")
        elif image_data[:6] in (b'GIF87a', b'GIF89a'):
            print("Detected GIF format")
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            print("Detected WebP format")
        else:
            print("Unknown or corrupted image format")
            # Log first 20 bytes for debugging
            print(f"First 20 bytes: {image_data[:20]}")
        
        # Try to open with Pillow
        try:
            image_stream = io.BytesIO(image_data)
            image = Image.open(image_stream)
            print(f"Successfully opened image: {image.format}, {image.size}, {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                print(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Save to BytesIO as PNG
            output_stream = io.BytesIO()
            image.save(output_stream, format='PNG')
            output_stream.seek(0)
            
            # Convert to base64
            base64_image = base64.b64encode(output_stream.getvalue()).decode('utf-8')
            
            print(f"Image processed successfully, base64 length: {len(base64_image)} characters")
            return base64_image
            
        except Exception as e:
            print(f"Pillow failed to process image: {e}")
            
            # Fallback: try to return the data as base64 if it's already valid image data
            try:
                base64_image = base64.b64encode(image_data).decode('utf-8')
                print(f"Fallback: returning raw image data as base64, length: {len(base64_image)}")
                return base64_image
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return None
        
    except Exception as e:
        print(f"Error processing image data: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_style_prompt(selected_items):
    """Create a detailed prompt for image generation based on selected fashion items"""
    
    # Extract details from selected items
    categories = list(set([item.get('category', 'N/A') for item in selected_items]))
    colors = list(set([item.get('color', 'N/A') for item in selected_items]))
    themes = list(set([item.get('theme', 'N/A') for item in selected_items]))
    
    # Build detailed outfit components
    outfit_components = "\n".join([
        f"- {item.get('name', 'Unknown Item')}: {item.get('description', 'No description.')}" 
        for item in selected_items
    ])
    
    # Create comprehensive prompt
    prompt = f"""
    Create a high-quality, professional fashion photograph of a complete styled outfit and showcase it on a woman model.

    OUTFIT COMPONENTS:
    {outfit_components}

    STYLE SPECIFICATIONS:
    - Categories: {', '.join(categories)}
    - Color Palette: {', '.join(colors)}
    - Style Themes: {', '.join(themes)}

    VISUAL REQUIREMENTS:
    - Fashion outfit should be shown on a woman model
    - Professional fashion photography studio setting
    - Clean, well-lit environment with neutral white/gray background
    - Model wearing the complete coordinated outfit
    - All specified items clearly visible and well-styled
    - High-resolution, sharp focus, editorial quality
    - Elegant model pose, sophisticated composition
    - Modern fashion magazine aesthetic
    - Portrait orientation (3:4 aspect ratio)

    Generate a stunning, cohesive fashionable image of a woman whose outfit combines all these elements into one perfectly styled look.
    """
    
    return prompt.strip()

def generate_image_with_gemini(prompt):
    """Generate image using Gemini API and return base64 data directly"""
    try:
        print(f"Generating image with prompt: {prompt[:100]}...")
        
        # Initialize Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        model = "gemini-2.0-flash-preview-image-generation"
        
        # Create content for the API
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        # Configure generation settings
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            response_mime_type="text/plain",
        )
        
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        
        # Generate content using streaming
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            
            # Check for image data
            if (chunk.candidates[0].content.parts[0].inline_data and 
                chunk.candidates[0].content.parts[0].inline_data.data):
                
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                data_buffer = inline_data.data
                mime_type = inline_data.mime_type
                
                print(f"Received image data from API - MIME: {mime_type}, Data type: {type(data_buffer)}")
                
                # Process image data directly in memory
                base64_image = process_image_data(data_buffer, mime_type)
                
                if base64_image:
                    # Store in memory
                    global recent_image_data
                    recent_image_data = {
                        'base64': base64_image,
                        'mime_type': 'image/png',
                        'timestamp': datetime.now().isoformat(),
                        'image_id': image_id
                    }
                    
                    print(f"Image generated successfully: {image_id}")
                    return {
                        'image_base64': base64_image,
                        'image_id': image_id,
                        'mime_type': 'image/png'
                    }
                else:
                    print("Failed to process image data")
                    return None
            else:
                # Print any text output
                if hasattr(chunk, 'text') and chunk.text:
                    print(f"API Response: {chunk.text}")
        
        print("No image was generated")
        return None
            
    except Exception as e:
        print(f"Error generating image with Gemini: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/generate-style', methods=['POST'])
def generate_style():
    """
    Main route to generate style image based on selected fashion items
    Expects JSON payload with selectedItems array
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'selectedItems' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request: No selectedItems provided.'
            }), 400

        selected_items = data['selectedItems']
        
        if not isinstance(selected_items, list) or not selected_items:
            return jsonify({
                'success': False,
                'error': 'Invalid request: selectedItems must be a non-empty array.'
            }), 400

        print(f"Received {len(selected_items)} items for style generation.")

        # Create the prompt for AI
        style_prompt = create_style_prompt(selected_items)
        print("This is the prompt===============================================================================================")
        print(style_prompt)
        print(f"Generated prompt length: {len(style_prompt)} characters")

        # Generate image using Gemini API
        image_result = generate_image_with_gemini(style_prompt)

        if not image_result:
            return jsonify({
                'success': False,
                'error': 'Failed to generate image with AI. Please try again.'
            }), 500

        # Create success response with image data
        response_data = {
            'success': True,
            'message': 'Style generated successfully!',
            'generated_description': f"AI-generated outfit combining {len(selected_items)} selected fashion items into a cohesive, stylish look.",
            'style_analysis': {
                'total_items': len(selected_items),
                'categories': list(set(item.get('category') for item in selected_items if item.get('category'))),
                'dominant_colors': list(set(item.get('color') for item in selected_items if item.get('color'))),
                'style_themes': list(set(item.get('theme') for item in selected_items if item.get('theme'))),
                'items_used': [item.get('name') for item in selected_items if item.get('name')]
            },
            'timestamp': datetime.now().isoformat(),
            'image_data': {
                'base64': image_result['image_base64'],
                'image_id': image_result['image_id'],
                'mime_type': image_result['mime_type']
            }
        }

        print("Successfully generated and processed image. Sending response.")
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error in /generate-style endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/recent-image', methods=['GET'])
def serve_recent_image():
    """Serve the most recent generated image from memory"""
    try:
        global recent_image_data
        
        if recent_image_data['base64'] is None:
            return jsonify({
                'success': False,
                'error': 'No recent image available.'
            }), 404
        
        return jsonify({
            'success': True,
            'image_data': {
                'base64': recent_image_data['base64'],
                'mime_type': recent_image_data['mime_type'],
                'image_id': recent_image_data['image_id'],
                'timestamp': recent_image_data['timestamp']
            }
        }), 200
        
    except Exception as e:
        print(f"Error serving recent image: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to serve recent image.'
        }), 500

@app.route('/generated-image/<image_id>', methods=['GET'])
def serve_generated_image(image_id):
    """Serve generated images by ID from memory"""
    try:
        global recent_image_data
        
        # Check if the requested image ID matches the recent image
        if recent_image_data['image_id'] == image_id and recent_image_data['base64'] is not None:
            return jsonify({
                'success': True,
                'image_data': {
                    'base64': recent_image_data['base64'],
                    'mime_type': recent_image_data['mime_type'],
                    'image_id': recent_image_data['image_id'],
                    'timestamp': recent_image_data['timestamp']
                }
            }), 200
        
        return jsonify({
            'success': False,
            'error': 'Image not found.'
        }), 404

    except Exception as e:
        print(f"Error serving image {image_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to serve image.'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global recent_image_data
    
    return jsonify({
        'status': 'healthy',
        'message': 'StyleAI Backend with Gemini API is running (No file saving)',
        'timestamp': datetime.now().isoformat(),
        'api_configured': bool(os.environ.get("GEMINI_API_KEY")),
        'model': 'gemini-2.0-flash-preview-image-generation',
        'recent_image_available': recent_image_data['base64'] is not None,
        'recent_image_timestamp': recent_image_data['timestamp']
    }), 200

if __name__ == '__main__':
    # Check API key configuration
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key before running the server.")
        print("Example: export GEMINI_API_KEY='your_api_key_here'")
    else:
        print("✅ Gemini API key configured")
    
    print("Starting StyleAI Flask Backend (No File Saving)...")
    print("Model: gemini-2.0-flash-preview-image-generation")
    print("Images will be processed in memory and sent directly to frontend")
    print("Server will run on: http://localhost:5000")
    
    # Run the Flask app
    app.run(host='0.0.0.0')

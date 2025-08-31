import re
import numpy as np
from urllib.parse import urlparse
from PIL import Image
import cv2
import tldextract

def extract_url_features(url):
    """
    Extract 31 features from a URL for phishing detection
    Based on common URL-based phishing detection features
    """
    try:
        # Parse URL
        parsed = urlparse(url)
        domain_info = tldextract.extract(url)
        
        # Initialize features list
        features = []
        
        # 1. Length of URL
        features.append(len(url))
        
        # 2. Length of hostname
        features.append(len(parsed.netloc))
        
        # 3. IP address presence (1 if IP, 0 if not)
        ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        features.append(1 if ip_pattern.search(parsed.netloc) else 0)
        
        # 4. Number of dots
        features.append(url.count('.'))
        
        # 5. Number of hyphens
        features.append(url.count('-'))
        
        # 6. Number of @ symbols
        features.append(url.count('@'))
        
        # 7. Number of question marks
        features.append(url.count('?'))
        
        # 8. Number of & symbols
        features.append(url.count('&'))
        
        # 9. Number of | symbols
        features.append(url.count('|'))
        
        # 10. Number of = symbols
        features.append(url.count('='))
        
        # 11. Number of underscores
        features.append(url.count('_'))
        
        # 12. Number of tildes
        features.append(url.count('~'))
        
        # 13. Number of percent symbols
        features.append(url.count('%'))
        
        # 14. Number of slashes
        features.append(url.count('/'))
        
        # 15. Number of asterisks
        features.append(url.count('*'))
        
        # 16. Number of colons
        features.append(url.count(':'))
        
        # 17. Number of commas
        features.append(url.count(','))
        
        # 18. Number of semicolons
        features.append(url.count(';'))
        
        # 19. Number of dollar signs
        features.append(url.count('$'))
        
        # 20. Number of spaces
        features.append(url.count(' '))
        
        # 21. Number of 'www' occurrences
        features.append(url.lower().count('www'))
        
        # 22. Number of 'com' occurrences
        features.append(url.lower().count('com'))
        
        # 23. Number of double slashes
        features.append(url.count('//'))
        
        # 24. HTTP in path (excluding protocol)
        path_query = parsed.path + parsed.query
        features.append(1 if 'http' in path_query.lower() else 0)
        
        # 25. HTTPS token in URL (excluding protocol)
        features.append(1 if 'https' in url.lower().replace('https://', '') else 0)
        
        # 26. Ratio of digits in URL
        digits = sum(c.isdigit() for c in url)
        features.append(digits / len(url) if len(url) > 0 else 0)
        
        # 27. Ratio of digits in hostname
        hostname_digits = sum(c.isdigit() for c in parsed.netloc)
        features.append(hostname_digits / len(parsed.netloc) if len(parsed.netloc) > 0 else 0)
        
        # 28. Punycode presence
        features.append(1 if 'xn--' in parsed.netloc else 0)
        
        # 29. Port presence (1 if non-standard port, 0 if standard or no port)
        port = parsed.port
        standard_ports = [80, 443]
        features.append(1 if port and port not in standard_ports else 0)
        
        # 30. TLD in path
        tld = domain_info.suffix
        features.append(1 if tld and tld in parsed.path else 0)
        
        # 31. TLD in subdomain
        subdomain = domain_info.subdomain
        features.append(1 if tld and subdomain and tld in subdomain else 0)
        
        return features
        
    except Exception as e:
        # Return default features if extraction fails
        return [0] * 31

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for CNN model
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def extract_visual_features(image_path):
    """
    Extract visual features from screenshot for analysis
    """
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        
        if img is None:
            raise Exception("Could not load image")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate basic visual features
        features = {
            'mean_brightness': np.mean(img_rgb),
            'std_brightness': np.std(img_rgb),
            'dominant_colors': extract_dominant_colors(img_rgb),
            'edge_density': calculate_edge_density(img),
            'color_histogram': calculate_color_histogram(img_rgb)
        }
        
        return features
        
    except Exception as e:
        raise Exception(f"Error extracting visual features: {str(e)}")

def extract_dominant_colors(image, k=5):
    """Extract dominant colors from image"""
    try:
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_
        
        # Calculate percentages
        labels = kmeans.labels_
        percentages = [np.sum(labels == i) / len(labels) for i in range(k)]
        
        return list(zip(colors.tolist(), percentages))
        
    except Exception:
        return []

def calculate_edge_density(image):
    """Calculate edge density in image"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        return float(edge_density)
        
    except Exception:
        return 0.0

def calculate_color_histogram(image, bins=32):
    """Calculate color histogram for image"""
    try:
        # Calculate histogram for each channel
        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        hist_r = hist_r.flatten() / np.sum(hist_r)
        hist_g = hist_g.flatten() / np.sum(hist_g)
        hist_b = hist_b.flatten() / np.sum(hist_b)
        
        return {
            'red': hist_r.tolist(),
            'green': hist_g.tolist(),
            'blue': hist_b.tolist()
        }
        
    except Exception:
        return {'red': [], 'green': [], 'blue': []}

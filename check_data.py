import pandas as pd
import matplotlib.pyplot as plt

# 1. डेटा लोड करना (पाथ चेक कर लें)
try:
    # मान लेते हैं आपकी फाइल data फोल्डर में train.csv नाम से है
    df = pd.read_csv('data/train.csv') 
    print("✅ डेटा सफलतापूर्वक मिल गया!")
    print(f"कुल Rows: {df.shape[0]}, कुल Columns: {df.shape[1]}")
    
    # 2. पहली इमेज देखना (Visualization)
    # पहली कॉलम को छोड़कर बाकी पिक्सल्स हैं
    first_image = df.iloc[0, 1:].values.reshape(28, 28)
    
    plt.imshow(first_image, cmap='gray')
    plt.title(f"Label: {df.iloc[0, 0]}")
    plt.show()
    print("✅ विज़ुअलाइज़ेशन सफल रहा!")

except FileNotFoundError:
    print("❌ एरर: 'data/train.csv' फाइल नहीं मिली। कृपया चेक करें कि फाइल सही फोल्डर में है।")
    
# 🏗️ DraftWalk

**DraftWalk** is an intelligent architectural visualization tool that transforms 2D floor plan images into interactive 3D environments. By leveraging computer vision and OCR, it automates the tedious process of 3D modeling from sketches or blueprints.

---

## 🌟 Key Features

- **Automated Modeling**: Converts 2D image uploads directly into 3D walls, floors, and objects.
- **Intelligent Scene Analysis**:
    - **Wall Detection**: Directional morphological analysis to extract accurate wall structures.
    - **Thickness-Based Classification**: Separates structural walls from furniture outlines and annotations based on line weight.
    - **Door & Window Identification**: Uses wall-gap analysis and perimeter sensing (Convex Hull) to distinguish between interior doors and exterior windows.
    - **Staircase Recognition**: Pattern-matching for parallel lines to render 3D step geometry.
- **AI Assistant**: A built-in chat interface to help you modify and navigate the scene.
- **OCR Room Labeling**: Automatically identifies room types (Kitchen, Bedroom, etc.) and populates them with relevant furniture heuristics.
- **Modern UI**: Clean, responsive dashboard with a dedicated 3D Viewport, Left Management Sidebar, and Right Chat Sidebar.

---

## 🛠️ Tech Stack

### Frontend
- **React 18** + **Vite**
- **Three.js** (via **React Three Fiber**)
- **Zustand** (State Management)
- **Vanilla CSS** (Custom Design System)

### Backend
- **FastAPI** (Python)
- **OpenCV** (Image Processing)
- **NumPy** (Numerical Analysis)
- **Tesseract OCR** (Text Recognition)

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- Tesseract OCR engine installed on your system.
    - *Linux (Arch)*: `sudo pacman -S tesseract tesseract-data-eng`
    - *Linux (Ubuntu)*: `sudo apt install tesseract-ocr`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yash-217/DraftWalk.git
   cd DraftWalk
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # or activate.fish for fish users
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## 📐 How it Works (The Pipeline)

1. **Binarization**: The image is processed using Otsu's thresholding to isolate lines.
2. **Feature Separation**: A Distance Transform is applied to distinguish "thick" walls from "thin" furniture/annotations.
3. **Wall Extraction**: Skeletonization and Probabilistic Hough Transform extract the core wall segments.
4. **Perimeter Analysis**: The engine computes a Convex Hull of the floor plan to identify which walls are "external."
5. **Gap Analysis**:
    - Gaps in **external** walls are rendered as **Windows** (transparent glass panels).
    - Gaps in **internal** walls with a detected arc are rendered as **Doors**.
6. **Room Classification**: OCR reads room labels. If labels are missing, rooms are classified by relative area heuristics.
7. **Heuristic Placement**: Furniture is intelligently guessed based on the room type (e.g., placing a Bed in a detected Bedroom).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

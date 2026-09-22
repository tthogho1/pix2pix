package main

import (
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"os"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"
)

var imageExts = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".bmp":  true,
}

func isImageFile(name string) bool {
	return imageExts[strings.ToLower(filepath.Ext(name))]
}

func processImage(inPath, outPath string, imgSize, outImgWidth, outImgHeight, edgeThresh int) error {
	// 画像のリサイズ
	resizeMat := gocv.IMRead(inPath, gocv.IMReadAnyColor)
	if resizeMat.Empty() {
		return fmt.Errorf("failed to read image: %s", inPath)
	}
	defer resizeMat.Close()

	gocv.Resize(resizeMat, &resizeMat, image.Point{imgSize, imgSize}, 0, 0, gocv.InterpolationDefault)

	// グレースケールさせてからエッジ抽出
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(resizeMat, &gray, gocv.ColorBGRToGray)

	edge := gocv.NewMat()
	defer edge.Close()
	gocv.Canny(gray, &edge, float32(edgeThresh), float32(edgeThresh*3))

	// 画像の結合　左：リサイズ　右：エッジ抽出
	outImg := image.NewRGBA(image.Rect(0, 0, outImgWidth, outImgHeight))

	resizeImg, err := resizeMat.ToImage()
	if err != nil {
		return err
	}
	rectLeft := image.Rect(0, 0, imgSize, imgSize)
	draw.Draw(outImg, rectLeft, resizeImg, image.Point{0, 0}, draw.Over)

	edgeImg, err := edge.ToImage()
	if err != nil {
		return err
	}
	rectRight := image.Rect(imgSize, 0, imgSize+imgSize, imgSize)
	draw.Draw(outImg, rectRight, edgeImg, image.Point{0, 0}, draw.Over)

	// 出力先のサブフォルダを作成
	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		return err
	}

	// 画像の保存
	outFile, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	qt := jpeg.Options{Quality: 100}
	return jpeg.Encode(outFile, outImg, &qt)
}

func main() {
	inputdir := "./images/"
	out := "./outdata/"

	imgSize := 256
	outImgWidth := 512
	outImgHeight := 256

	// エッジ抽出の閾値
	edgeThresh := 40

	fmt.Println("image convert start")

	err := filepath.Walk(inputdir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if !isImageFile(info.Name()) {
			return nil
		}

		relPath, err := filepath.Rel(inputdir, path)
		if err != nil {
			return err
		}
		outPath := filepath.Join(out, relPath)

		fmt.Println("processing:", relPath)
		if procErr := processImage(path, outPath, imgSize, outImgWidth, outImgHeight, edgeThresh); procErr != nil {
			fmt.Fprintln(os.Stderr, procErr)
		}
		return nil
	})

	if err != nil {
		panic(err)
	}

	fmt.Println("image convert end")
}

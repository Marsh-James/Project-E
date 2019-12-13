package uk.ac.soton.ecs.compvisgroup.apple.run1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * @author Oscar van Leusen
 */
class TinyImageProcessor implements FeatureExtractor<DoubleFV, FImage> {

    private int tinySize;

    TinyImageProcessor(int tinySize) {
        this.tinySize = tinySize;
    }

    /**
     * Generates a vector from the image using TinyImage ready for KNN.
     * This will have zero mean and be normalised to unit length.
     * @param image : Image to calculate TinyImage and then convert to vector
     * @return
     */
    @Override
    public DoubleFV extractFeature(FImage image) {
        FImage tinyImage = generateTinyImage(image);
        // Calculate the pixel mean, this can be subtracted from every pixel to zero-mean the Tiny Image
        float pixelMean = tinyImage.sum() / (float) Math.pow(tinySize, 2);
        FImage zeroMeanTinyImage = tinyImage.subtract(pixelMean);

        // Flatten the image to a vector and then normalise it to unit length.
        return new DoubleFV(zeroMeanTinyImage.getDoublePixelVector()).normaliseFV();
    }

    /**
     * Take a regular FImage and make a tiny version of it, according to the tinySize provided in the Constructor
     * @param image Regular Image
     * @return Tiny Image
     */
    private FImage generateTinyImage(FImage image) {
        // Since we're cropping to a square, take the smallest size as the new dimension.
        int size = Math.min(image.height, image.width);
        FImage squareImg = image.extractCenter(size, size);
        // Resize to a fixed resolution
        FImage tinyImage = squareImg.process(new ResizeProcessor(this.tinySize, this.tinySize));

        return tinyImage;
    }


}

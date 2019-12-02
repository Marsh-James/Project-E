package uk.ac.soton.ecs.compvisgroup.apple.run1;

import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.util.array.ArrayUtils;

import java.util.Arrays;

/**
 * @author Oscar van Leusen
 */
public class TinyImageProcessor {

    private int tinySize;

    public TinyImageProcessor(int tinySize) {
        this.tinySize = tinySize;
    }

    FImage generateTinyImage(FImage image) {
        // Since we're cropping to a square, take the smallest size as the new dimension.
        int size = Math.min(image.height, image.width);
        FImage squareImg = image.extractCenter(size, size);

        // Resize to a fixed resolution
        FImage tinyImage = squareImg.process(new ResizeProcessor(this.tinySize, this.tinySize));

        double[] test = tinyImage.getDoublePixelVector();


        ArrayUtils.reshape(tinyImage.pixels);



        System.out.println(Arrays.toString(test));
        return null;
    }
}

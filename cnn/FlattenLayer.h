#pragma once

#include <stdexcept>
#include <vector>

#include "Layer.h"

/// @class FlattenLayer
/// @brief A class representing a flattening layer in a neural network.
///
class FlattenLayer : public Layer
{
  public:
    /// @brief Default constructor for the FlattenLayer class.
    FlattenLayer() = default;

    /// @brief Performs the forward pass of the flatten layer.
    /// This method takes a 3D input tensor and flattens it into a 3D output tensor
    /// with a single batch dimension.
    ///
    /// @param [in] input A 3D tensor representing the input data to the layer,
    ///                  with dimensions (channels, height, width).
    /// @return A 3D tensor representing the output data after flattening,
    ///         with dimensions (1, 1, channels * height * width).
    /// @throws std::invalid_argument If the input tensor is empty or has an incorrect shape.
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        // Check if input is empty
        if (input.empty() || input[0].empty() || input[0][0].empty())
        {
            throw std::invalid_argument("Input cannot be empty or incorrectly shaped.");
        }

        // Get dimensions of the input
        size_t channels = input.size();       // Number of channels (e.g., RGB)
        size_t height   = input[0].size();    // Height of the input
        size_t width    = input[0][0].size(); // Width of the input

        // Persist original shape for backward reshape
        originalChannels_ = channels;
        originalHeight_   = height;
        originalWidth_    = width;

        size_t flattenedSize = channels * height * width;

        // Create the output tensor with one batch dimension
        std::vector<std::vector<std::vector<float>>> output(
            1,
            std::vector<std::vector<float>>(1, std::vector<float>(flattenedSize))); // Output shape: 1 x 1 x (C * H * W)

        // Flatten the input tensor
        size_t index = 0;                     // Index for the output tensor
        for (size_t c = 0; c < channels; ++c) // Loop over channels
        {
            for (size_t h = 0; h < height; ++h) // Loop over height
            {
                for (size_t w = 0; w < width; ++w) // Loop over width
                {
                    output[0][0][index++] = input[c][h][w]; // Assign flattened values to output tensor
                }
            }
        }

        return output; // Return the flattened output
    }

    /// @brief Performs the backward pass of the flatten layer.
    /// This method takes a 3D upstream gradient tensor and reshapes it back to
    /// the original input tensor dimensions for the previous layer in the network.
    ///
    /// @param [in] upstreamGradient A 3D tensor representing the gradient of the loss
    ///                              with respect to the output of this layer,
    ///                              with dimensions (number of samples, 1, channels * height * width).
    /// @param [in] learningRate The learning rate for any potential updates during backpropagation.
    /// @return A 3D tensor representing the gradient of the loss with respect to
    ///         the input of this layer, with dimensions (number of samples, channels, height, width).
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& upstreamGradient, float learningRate) override
    {
        if (originalChannels_ == 0 || originalHeight_ == 0 || originalWidth_ == 0)
        {
            throw std::runtime_error("FlattenLayer backward called without a stored forward shape.");
        }

        // Flatten upstream gradient into a single vector
        std::vector<float> flatGradients;
        flatGradients.reserve(originalChannels_ * originalHeight_ * originalWidth_);

        if (upstreamGradient.size() == 1 && upstreamGradient[0].size() == 1)
        {
            flatGradients.insert(flatGradients.end(), upstreamGradient[0][0].begin(), upstreamGradient[0][0].end());
        }
        else
        {
            for (const auto& channel : upstreamGradient)
            {
                for (const auto& row : channel)
                {
                    flatGradients.insert(flatGradients.end(), row.begin(), row.end());
                }
            }
        }

        const size_t expectedSize = originalChannels_ * originalHeight_ * originalWidth_;
        if (flatGradients.size() != expectedSize)
        {
            throw std::invalid_argument("Flattened gradient size does not match original tensor shape.");
        }

        // Reshape back to original 3D shape (no batch dimension in this implementation)
        std::vector<std::vector<std::vector<float>>> output(
            originalChannels_,
            std::vector<std::vector<float>>(originalHeight_, std::vector<float>(originalWidth_, 0.0f)));

        size_t index = 0;
        for (size_t c = 0; c < originalChannels_; ++c)
        {
            for (size_t h = 0; h < originalHeight_; ++h)
            {
                for (size_t w = 0; w < originalWidth_; ++w)
                {
                    output[c][h][w] = flatGradients[index++];
                }
            }
        }

        return output;
    }

    /// @brief Updates the weights of the flatten layer.
    /// This method is not applicable for the FlattenLayer as it does not have weights to update.
    ///
    /// @param [in] learningRate The learning rate for adjusting the weights during the update process.
    void UpdateWeights(float learningRate) override
    {
        // FlattenLayer does not have weights, so this function does nothing
    }

    std::vector<float> GetAllWeights() const override
    {
        return {};
    }

    float ComputeL2Regularization(float l2RegularizationFactor) const
    {
        return 0;
    }

  private:
    size_t originalChannels_ = 0;
    size_t originalHeight_   = 0;
    size_t originalWidth_    = 0;
};

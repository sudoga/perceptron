import Control.Monad (forM_, when)
import Data.Csv (FromNamedRecord(..), (.:))
import Data.List (transpose)
import Data.Vector (Vector)
import System.Random (StdGen, newStdGen, randomRs)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.Vector as V

data Neuron = Neuron
  { activationFunction :: Double -> Double
  , threshold :: Double
  , weights :: [Double]
  , thresholdWeight :: Double
  , learningRate :: Double
  , epochs :: Int
  }

data TrainingData = TrainingData
  { trainX :: Vector (Vector Double)
  , trainY :: Vector Double
  }

instance FromNamedRecord TrainingData where
  parseNamedRecord r = do
    x1 <- r .: "x1"
    x2 <- r .: "x2"
    x3 <- r .: "x3"
    x4 <- r .: "x4"
    y <- r .: "y"
    return $ TrainingData (V.fromList [x1, x2, x3, x4]) y

sigmoidBipolar :: Double -> Double
sigmoidBipolar x
  | x < 0 = -1
  | x > 0 = 1
  | otherwise = 0

initializeNeuron :: StdGen -> (Double -> Double) -> Double -> Neuron
initializeNeuron gen activationFunc thresholdValue = Neuron
  { activationFunction = activationFunc
  , threshold = thresholdValue
  , weights = take 4 (randomRs (0, 1) gen)
  , thresholdWeight = head (randomRs (0, 1) gen)
  , learningRate = head (randomRs (0, 1) gen)
  , epochs = 0
  }

updateWeights :: Neuron -> Double -> Double -> Vector Double -> Neuron
updateWeights neuron@Neuron {..} expected result inputs = neuron
  { weights = zipWith (\w x -> w + learningRate * (expected - result) * x) weights (V.toList inputs)
  , thresholdWeight = thresholdWeight + learningRate * (expected - result) * threshold
  }

predict :: Neuron -> Vector (Vector Double) -> Vector Double
predict Neuron {..} inputs = V.map (\row -> activationFunction result)
  where
    result = sum (zipWith (*) (V.toList row) weights) + (threshold * thresholdWeight)

train :: Neuron -> TrainingData -> Int -> Bool -> IO Neuron
train neuron (TrainingData inputs outputs) maxEpochs verbose = go neuron 0
  where
    go neuron epoch
      | epoch == maxEpochs || V.null errors = return neuron
      | otherwise = do
          when verbose (putStrLn $ "Epoch: " ++ show epoch ++ " | Errors: " ++ show (V.length errors) ++ " | Accuracy: " ++ show accuracy)
          go updatedNeuron (epoch + 1)
      where
        predictions = predict neuron inputs
        errors = V.filter (\(pred, out) -> pred /= out) (V.zip predictions outputs)
        accuracy = 1 - (fromIntegral (V.length errors) / fromIntegral (V.length outputs))
        updatedNeuron = V.foldl' (\n (pred, out, input) -> updateWeights n out pred input) neuron errors
        trainNeuron :: Neuron -> TrainingData -> Int -> IO Neuron
trainNeuron neuron trainingData maxEpochs = train neuron trainingData maxEpochs False

main :: IO ()
main = do
  csvData <- BL.readFile "training_data.csv"
  case Csv.decodeByName csvData of
    Left err -> putStrLn err
    Right (_, v) -> do
      let trainingData = V.map (\(TrainingData x y) -> TrainingData (V.fromList [x]) y) v
      let trainInputs = V.map trainX trainingData
      let trainOutputs = V.map trainY trainingData

      gen <- newStdGen
      let neuron = initializeNeuron gen sigmoidBipolar (-1)

      trainedNeuron <- trainNeuron neuron (TrainingData trainInputs trainOutputs) 300
      putStrLn $ "Trained weights: " ++ show (weights trainedNeuron)
      putStrLn $ "Trained threshold weight: " ++ show (thresholdWeight trainedNeuron)

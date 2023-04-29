import System.Random
import Data.List
import Data.Csv
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V

data Neuron = Neuron
  { fg           :: Double -> Int
  , fixedValue   :: Double
  , teta         :: Double
  , wteta        :: Double
  , wi           :: [Double]
  , learningRate :: Double
  , epoch        :: Int
  } deriving Show

type DataFrame = [([Double], Double)]

randomList :: Int -> IO [Double]
randomList 0 = return []
randomList n = do
  r  <- randomRIO (0, 1)
  rs <- randomList (n-1)
  return (r:rs)

sigmoidBipolar :: Double -> Int
sigmoidBipolar x
  | x < 0     = -1
  | x > 0     = 1
  | otherwise = 0

initialNeuron :: (Double -> Int) -> Double -> IO Neuron
initialNeuron fgFunction tetaValue = do
  wTeta <- randomRIO (0, 1)
  return $ Neuron fgFunction 0.0 tetaValue wTeta [] 0.0 0

train :: Neuron -> DataFrame -> Int -> Double -> Bool -> IO Neuron
train neuron@Neuron{..} trainData maxEpochs learningRate verbose = do
  let numFeatures = length (fst $ head trainData)
  randomW <- randomList numFeatures
  let initialWi = randomW
  trainLoop neuron{wi = randomW, learningRate = learningRate, epoch = 0} trainData maxEpochs verbose

trainLoop :: Neuron -> DataFrame -> Int -> Bool -> IO Neuron
trainLoop neuron@Neuron{..} trainData maxEpochs verbose
  | epoch >= maxEpochs = return neuron
  | otherwise = do
    let (newWi, newWTeta) = updateWi learningRate (zip trainData (map (predict neuron) trainData))
    let newNeuron = neuron{wi = newWi, wteta = newWTeta, epoch = epoch + 1}
    if verbose
      then putStrLn ("Epoch: " ++ show epoch ++ " | Errors: " ++ show (errors trainData (map (predict newNeuron) trainData)) ++ " | Accuracy: " ++ show (accuracy trainData (map (predict newNeuron) trainData))) >> trainLoop newNeuron trainData maxEpochs verbose
      else trainLoop newNeuron trainData maxEpochs verbose

updateWi :: Double -> [(([Double], Double), Int)] -> ([Double], Double)
updateWi learningRate trainDataPredictions = (updatedWi, updatedWTeta)
  where
    wiDeltas = [learningRate * (expected - result) * input | ((input, expected), result) <- trainDataPredictions, input <- [0 .. length (fst $ head trainDataPredictions) - 1]]
    updatedWi = map sum . transpose $ chunksOf (length trainDataPredictions) wiDeltas
    wtetaDelta = sum [learningRate * (expected - result) * teta | ((_ , expected), result) <- trainDataPredictions]
    updatedWTeta = wteta + wtetaDelta

predict :: Neuron -> [Double] -> Int
predict Neuron{..} xs = fg $ sum (zipWith (*) wi xs) + teta * wteta

errors :: DataFrame -> [Int] -> Int
errors trainData predictions = length $ filter (uncurry (/=)) $ zip (map snd trainData) predictions

accuracy :: DataFrame -> [Int] -> Double
accuracy trainData predictions = 1 - fromIntegral (errors trainData predictions) / fromIntegral (length trainData)

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

main :: IO ()
main = do
  let trainData = [ ([1, 1, 1, 1], 1)
                  , ([2, 2, 2, 2], 2)
                  , ([3, 3, 3, 3], 3)
                  , ([4, 4, 4, 4], 4)
                  ]
  neuron <- initialNeuron sigmoidBipolar (-1)
  trainedNeuron <- train neuron trainData 300 0.5 True
  putStrLn "Trained Neuron:"
  print trainedNeuron

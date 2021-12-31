# Reading Text

<ul><li><b>Initial Probability</b> - The ratio of the number of times single character appears in the first position of the word to the total occurances of the character in the data is known as initial probability.<br><br>
</li><li><b>Transition Probability</b> - It is defined as the ratio of the probability of one character coming after another character in the data divided by the probability of the occurence of total number of characters in the given training data file. <br><br>
</li><li><b>Emission Probability</b> - In this particular problem we are calculating the emission probability by comparing the pixels in the test character to the pixels of the train character. Then the number of matched star values(black pixels), the number of matched space characters(white pixels) and the number of unmatched characters are assigned weights as 0.7,0.2 and 0.1 respectively. This determines if a given test character is a specific train character or not and by how much probability. These probabilities are stored in a dictionary and are used as emission probabilities.<br><br>
</li></ul>

<h3> 1. Simple Bayes Net </h3> <br>
For Bayes net, the general approach is to use the maximum emission probability and the prior probability of the character. In this approach, we can skip the prior probabilities as we are calculating the emission probabilities by pixel comparison. So, the character is directly predicted as the maximum value of the emission probability<br>

<h3> 2. Viterbi </h3> <br>
<ul><li>The viterbi table is created using the emission probabilities calculated by pixel comparison, the initial probabilities of the characters as the appear in the training text file and the transition probabilities of the characters, again, calculated from the training text file.</li>
<li>The path of the characters are stored in the table as the table is being created.</li>
<li>This path is then back tracked based on the max value of the probability calculated at the last level of the viterbi table.</li>
<li>The returned string is then reversed to get the correct order of the characters.</li></ul><br>

The challenges we faced in implementing the Viterbi algorithm for this problem were as follows:-<br>
<ul><li>When only the pixels that were matching were considered to calculate the emission probabilities, the character prediction for viterbi was very poor. To improve this, we assigned weights to the matched and unmatched pixels.</li>
<li>After assigning weights to matched and unmatched pixels, it was still giving a poor output for the first character of the test images as compared to the Bayes Net model. This was improved by assigning a weight to the emission probabilities for the first character in the viterbi table. This improved the prediction of first character for viterbi for 6 images.</li>

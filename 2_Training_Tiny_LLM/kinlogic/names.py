"""Name pools for generated questions: Indian (the exams' usual names) and international, by gender."""

INDIAN_M = '''Aarav Abhay Abhishek Aditya Ajay Akash Akhil Alok Aman Amar Amit Anand Anil Ankit Ankur Anuj Anupam Arjun
Arun Ashok Ashwin Atul Ayush Balaji Bharat Bhaskar Chandan Chetan Darshan Deepak Dev Dhruv Dilip Dinesh Gaurav Girish
Gopal Govind Hari Harish Harsh Hemant Hitesh Ishaan Jagdish Jatin Jay Jitendra Kabir Kailash Kamal Karan Kartik
Kishore Krishna Kunal Lalit Lokesh Madhav Mahesh Manish Manoj Mayank Mehul Mohan Mohit Mukesh Murali Naveen Neeraj
Nikhil Nilesh Nitin Om Pankaj Paras Parth Pawan Prakash Pranav Prasad Pratap Praveen Prem Puneet Rahul Raj Rajat
Rajesh Rakesh Ram Ramesh Ravi Rohan Rohit Sachin Sagar Sahil Sameer Sandeep Sanjay Santosh Satish Shankar Shekhar
Shiv Shyam Siddharth Sohan Subhash Sudhir Sumit Sunil Suraj Suresh Tarun Tushar Uday Umesh Varun Vijay Vikas Vikram
Vinay Vinod Vishal Vivek Yash Yogesh Dinabandhu Bajpai Harshit Kushal Pradeep Rajiv Sushant Tanmay Utkarsh Vasant
Gautam Nakul Sahdev Bhim Arnav Advait Vihaan Reyansh Rudra Shaurya Kiaan Atharv Divyansh Yuvraj Hrithik Ritesh'''.split()
INDIAN_F = '''Aarti Aditi Aisha Akshara Amrita Ananya Anita Anjali Ankita Anu Anuradha Aparna Archana Asha Bhavna Charu
Deepa Deepika Divya Durga Gauri Geeta Gita Heena Hema Indu Isha Jaya Jyoti Kajal Kalpana Kamala Kavita Kiran Kirti
Komal Kriti Lakshmi Lata Leela Madhu Mahima Malini Mamta Manisha Maya Meena Meera Megha Mira Mona Nandini Neelam Neha
Nidhi Nikita Nisha Pallavi Pooja Poonam Prachi Pragya Pratibha Preeti Priya Radha Rani Rashmi Rekha Renu Ritu Riya
Roshni Ruchi Rupa Sakshi Sangeeta Sanghoti Sapna Sarita Seema Shalini Shanti Sheetal Shilpa Shreya Shruti Shweta Simran
Sita Smita Sneha Sonal Sonia Sudha Suman Sunita Swati Tanvi Tara Uma Usha Vandana Varsha Vidya Vimla Yamini Zoya
Anushka Ira Kavya Myra Saanvi Diya Navya Riddhi Siddhi Tanya Trisha Vaishnavi Pihu Aadya Kashvi Gunjan Hemlata Ritika'''.split()
WORLD_M = '''Aaron Adam Adrian Alan Albert Alex Andrew Anthony Arthur Ben Brian Bruce Carl Charles Chris Daniel David
Dennis Derek Edward Eric Ethan Frank Gary George Harry Henry Jack Jacob James Jason Jeff John Joseph Kevin Kyle Leo
Liam Louis Luke Martin Matthew Michael Nathan Neil Noah Oliver Oscar Patrick Paul Peter Philip Ray Richard Robert
Ryan Sam Scott Simon Steven Thomas Tim Tom Tony Victor Walter William Ahmed Ali Omar Hassan Yusuf Kenji Hiroshi Wei
Chen Lars Sven Pablo Diego Mateo Luca Marco Andre Ivan Dmitri Kwame Kofi Tunde Felix Hugo Elias Jonas Mason Hunter'''.split()
WORLD_F = '''Abigail Alice Amanda Amelia Amy Angela Anna Ashley Barbara Betty Carol Catherine Chloe Claire Diana Donna
Dorothy Elena Elizabeth Ella Emily Emma Eva Fiona Grace Hannah Helen Isabel Jane Jennifer Jessica Julia Karen Kate
Laura Linda Lisa Lucy Margaret Maria Martha Mary Megan Mia Natalie Nancy Nina Olivia Paula Rachel Rebecca Rose Ruth
Sara Sarah Sophia Susan Tina Victoria Wendy Zoe Fatima Layla Mariam Yuki Mei Ling Ingrid Freya Lucia Sofia Giulia
Camila Valentina Olga Tatiana Ama Amara Nia Chiara Clara Lena Maya Iris Hope Will'''.split()
WORLD_F = [n for n in WORLD_F if n != 'Will']
SURNAMES = '''Sharma Verma Gupta Singh Kumar Mehta Patel Shah Reddy Rao Nair Iyer Menon Pillai Das Bose Ghosh Sen Roy
Mishra Pandey Tiwari Dubey Joshi Kapoor Malhotra Khanna Chopra Arora Bhatia Saxena Agarwal Jain Banerjee Mukherjee
Chatterjee Naidu Desai Kulkarni Patil Jadhav Smith Johnson Brown Taylor Wilson Clark Lewis Walker Hall Young'''.split()

# Letters used as people ("A is the brother of B"); I is the pronoun.
LETTERS = [c for c in 'ABCDEFGHJKLMNOPQRSTUVWXYZ']

# Gender of common names, for questions that never say it (exams assume "Dinabandhu" is a man).
MORE_M = '''Aadesh Aakash Aarush Abhinav Abhiram Adarsh Adil Advait Agastya Ahmed Ajit Akbar Akshay Alok Amitabh Amol Anant
Anirban Anirudh Ansh Anshul Anurag Apurva Arnav Arpit Arun Arvind Ashish Ashok Ashwin Atharva Atul Avinash Ayaan Ayush
Badal Balaji Balram Bharat Bhaskar Bhavesh Bhushan Bipin Brijesh Chaitanya Chandan Chetan Chirag Daksh Darshan Dev
Devansh Devendra Dhananjay Dharmesh Dhruv Dilip Dinesh Divyansh Eklavya Faisal Farhan Gagan Ganesh Gaurav Girish
Gopal Govind Gulshan Hardik Harish Harshad Hemant Himanshu Hitesh Imran Indrajit Irfan Ishaan Jagdish Jai Jatin Jayant
Jayesh Jitendra Kabir Kailash Kamal Kapil Karthik Kartik Kashyap Kedar Keshav Ketan Kishore Krishna Kunal Lakshman
Lalit Laxman Madhav Mahendra Mahesh Manav Manish Manoj Mayank Mihir Milind Mohit Mukesh Murali Nachiket Naman Narayan
Naresh Naveen Nayan Neeraj Nikhil Nilesh Niranjan Nishant Omkar Pankaj Paras Parth Pawan Piyush Pradeep Pramod
Pranav Prasad Prashant Pratap Praveen Prem Pritam Puneet Purushottam Raghav Raghu Rajat Rajendra Rajesh Rajiv Rakesh
Ramesh Ranjit Ravindra Rishabh Rishi Ritesh Rohan Rohit Ronak Rudra Sachin Sagar Sahil Sai Saket Salman Sameer
Sandeep Sanjay Santosh Sarthak Satish Saurabh Shailesh Shantanu Sharad Shaurya Shekhar Shiv Shivam Shreyas Shubham
Siddharth Sohail Somnath Subhash Sudhir Sujit Sumit Sunil Suraj Suresh Surya Swapnil Tanmay Tarun Tejas Tushar Uday
Umesh Utkarsh Vaibhav Varun Vasant Ved Vedant Venkat Vihaan Vijay Vikas Vikram Vinay Vinod Vipul Viraj Vishal Vishnu
Vivek Yash Yashwant Yogesh Yuvraj Zubair Zaheer Abdul Asif Aziz Bilal Danish Ehsan Fahad Hamid Hassan Iqbal Javed
Junaid Kareem Mansoor Mustafa Nadeem Rashid Rizwan Sajid Shahid Tariq Wasim Yusuf Jasbir Gurpreet Harpreet Manpreet
Jaswant Kuldeep Navjot Ranbir Sukhwinder Tejinder Arijit Debashish Partha Pranab Saurav Soumya Subhro Tapan Biswajit
Abhijit Anupam Jeevan Joseph Mathew Thomas Varghese Ajmal Rahim Selvam Murugan Senthil Karthikeyan Ilango Arumugam
Pandian Ramanathan Srinivas Raghunath Venkatesh Nagaraj Shankar Mallikarjun Basavaraj Prakash Raju Rohan Aniket'''.split()
MORE_F = '''Aadya Aanya Aarohi Abha Akanksha Alka Amrita Anamika Anandi Anika Anushka Anvi Aparajita Aradhya Arpita
Asmita Avni Ayesha Bhagyashree Bharti Bhavana Bindu Chandni Chhaya Damini Deepa Deepali Deepti Devika Dhanashree
Divya Durga Ekta Esha Farah Fatima Gargi Gayatri Geeta Gita Gunjan Hema Hemlata Indira Indu Ira Isha Ishita Jahnavi
Janaki Jasleen Jasmine Juhi Kajal Kalpana Kamala Kanchan Kanika Karishma Kavita Kavya Khushi Kirti Komal Kritika
Kusum Lakshmi Lata Leela Lavanya Madhuri Mahima Malati Malini Mamta Manisha Manju Mansi Maya Medha Meenakshi Megha
Mehak Minal Mitali Mona Mridula Mrunal Nalini Namita Nandini Navya Neelam Neeta Nidhi Niharika Nikita Nirmala Nisha
Nupur Padma Pallavi Pari Parvati Payal Pihu Pooja Poonam Prachi Pragya Pratibha Preeti Prerna Priya Priyanka Purnima
Radha Radhika Rajni Rakhi Rani Rashmi Rekha Renu Revati Rhea Richa Riddhi Rina Ritika Ritu Riya Rohini Ruchi Rupali
Sadhana Sakshi Saloni Sanchita Sangeeta Sanjana Sapna Sarita Saroj Savita Shalini Sharmila Sheetal Shilpa Shivani
Shobha Shraddha Shreya Shruti Shubhangi Simran Smita Sneha Sonal Sonali Sonam Suhani Sujata Sulochana Suman Sunanda
Sunita Supriya Surbhi Sushma Swara Swati Tanvi Tanya Tara Trisha Tulsi Uma Urmila Usha Vaishali Vandana Vani Varsha
Vasudha Vidya Vimala Vinita Yamini Zara Zoya Afreen Ameena Bushra Farzana Hina Mehrunnisa Nasreen Rukhsar Saba Sana
Shabana Shahnaz Tabassum Yasmin Gurleen Harleen Manjeet Navneet Simranjeet Ananya Debjani Ishani Moumita Paromita
Rituparna Sohini Sudeshna Tanushree Mary Teresa Annamma Lissy Ammini Kamakshi Meenal Selvi Kalyani Bhavani Janani
Sowmya Revathi Shanthi Vijaya Kokila Malathi Sukanya Hemavathi Lalitha Saraswati Tanu Pinky Chiku Minnie Guddi'''.split()
MORE_M += '''Salim Saleem Mohammed Mohammad Muhammad Mohd Ahmad Ali Hassan Hasan Hussain Husain Rashid Rahim Karim Faisal Zaid
Omar Umar Usman Bilal Hamza Tariq Nadeem Naveed Waseem Wasim Sohail Shoaib Asif Arif Aamir Salman Shahid Shakeel Junaid
Javed Jawed Riyaz Zaheer Mustafa Yasin Ibrahim Ismail Idris Anwar Akram Aslam Iqbal Jamal Kamal Khalid Majid Mansoor Nasir
Rafiq Rehan Sajid Sharif Zubair Yousuf Abdul Abdullah Rizwan Irshad Ejaz Feroz Firoz Altaf Afzal Azhar'''.split()
MORE_F += '''Zainab Lubna Shabnam Aisha Ayesha Khadija Mariam Maryam Nusrat Rubina Rukhsana Shazia Farida Hina Huma Iqra Mehreen
Mehwish Nazia Parveen Rehana Rizwana Sabiha Sadia Saira Salma Samina Shirin Sumaiya Yasmeen Zeba Zehra Nargis Naseem_no
Rabia Razia Sakina Shamim_no Tahira Zubeida Ruksar Asma Amina Fatimah Haseena Jameela Kausar Nafisa'''.split()
MORE_F = [n for n in MORE_F if not n.endswith('_no')]
UNISEX = {'kiran', 'harsha', 'shashi', 'sunny', 'jyoti', 'santoshi', 'rinku', 'bablu', 'pappu', 'guddu',   # never given a gender by name
          'gurpreet', 'harpreet', 'manpreet', 'jaspreet', 'manjeet', 'navneet', 'simranjeet', 'noor', 'shaheen', 'naseem', 'shamim'}
INDIAN_M = [n for n in dict.fromkeys(INDIAN_M + MORE_M) if n.lower() not in UNISEX]
INDIAN_F = [n for n in dict.fromkeys(INDIAN_F + MORE_F) if n.lower() not in UNISEX and n not in INDIAN_M]
NAME_GENDER = {**{x.lower(): 'M' for x in INDIAN_M + WORLD_M}, **{x.lower(): 'F' for x in INDIAN_F + WORLD_F}}


def name_priors(ents: list[dict]) -> dict:
    """{placeholder: gender} for names in prepare()'s table that the list knows."""
    return {e['id']: NAME_GENDER[e['text'].lower()] for e in ents if e['kind'] == 'name' and e['text'].lower() in NAME_GENDER}

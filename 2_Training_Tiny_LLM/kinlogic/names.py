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
NAME_GENDER = {**{x.lower(): 'M' for x in INDIAN_M + WORLD_M}, **{x.lower(): 'F' for x in INDIAN_F + WORLD_F}}


def name_priors(ents: list[dict]) -> dict:
    """{placeholder: gender} for names in prepare()'s table that the list knows."""
    return {e['id']: NAME_GENDER[e['text'].lower()] for e in ents if e['kind'] == 'name' and e['text'].lower() in NAME_GENDER}

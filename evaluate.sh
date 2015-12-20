
#python train.py training_data tempeval_QA.model

python predict.py test_data tempeval_QA.model tempeval_output

:<<'END'
Black_Death-en.ascii.TE3input.TE3input.tml          ebola_infection.ascii.TE3input.TE3input.tml         Renaissance-en.ascii.TE3input.TE3input.tml
blog1.TE3input.TE3input.tml                         Egypt-en.ascii.TE3input.TE3input.tml                Roman_Empire-en.ascii.TE3input.TE3input.tml
blog2.TE3input.TE3input.tml                         .gitignore                                          russian_pilgrims.ascii.TE3input.TE3input.tml
blog3.TE3input.TE3input.tml                         Great_Depression-en.ascii.TE3input.TE3input.tml     Sumer-en.ascii.TE3input.TE3input.tml
blog4.TE3input.TE3input.tml                         History_of_Paper-en.ascii.TE3input.TE3input.tml     US_Constitution-en.ascii.TE3input.TE3input.tml
blog5.TE3input.TE3input.tml                         james_brady_dies.ascii.TE3input.TE3input.tml        wall_st.ascii.TE3input.TE3input.tml
blog6.TE3input.TE3input.tml                         man_u_vs_liverpool.ascii.TE3input.TE3input.tml      wwi.ascii.TE3input.TE3input.tml
blog7.TE3input.TE3input.tml                         marijuana.ascii.TE3input.TE3input.tml               WWI-en.ascii.TE3input.TE3input.tml
blog8.TE3input.TE3input.tml                         microsoft_sues_samsung.ascii.TE3input.TE3input.tml  WWII-en.ascii.TE3input.TE3input.tml

END

mkdir tempeval_output/wikipedia-annotated-output
mkdir tempeval_output/blog-annotated-output
mkdir tempeval_output/news-annotated-output



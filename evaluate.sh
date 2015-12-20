
TEA_PATH=$(pwd)

python train.py training_data tempeval_QA.model

python predict.py test_data tempeval_QA.model tempeval_output

mkdir tempeval_output/wikipedia-annotated-output

mv tempeval_output/WWI-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/WWII-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/US_Constitution-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/Sumer-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/Roman_Empire-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/Renaissance-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/History_of_Paper-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/Great_Depression-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/Egypt-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/
mv tempeval_output/Black_Death-en.ascii.TE3input.tml tempeval_output/wikipedia-annotated-output/

mkdir tempeval_output/blog-annotated-output

mv tempeval_output/blog1.TE3input.tml tempeval_output/blog-annotated-output/
mv tempeval_output/blog2.TE3input.tml tempeval_output/blog-annotated-output/
mv tempeval_output/blog3.TE3input.tml tempeval_output/blog-annotated-output/
mv tempeval_output/blog4.TE3input.tml tempeval_output/blog-annotated-output/
mv tempeval_output/blog5.TE3input.tml tempeval_output/blog-annotated-output/
mv tempeval_output/blog6.TE3input.tml tempeval_output/blog-annotated-output/
mv tempeval_output/blog7.TE3input.tml tempeval_output/blog-annotated-output/
mv tempeval_output/blog8.TE3input.tml tempeval_output/blog-annotated-output/

mkdir tempeval_output/news-annotated-output

mv tempeval_output/pf_chang.ascii.TE3input.tml tempeval_output/news-annotated-output/
mv tempeval_output/wall_st.ascii.TE3input.tml  tempeval_output/news-annotated-output/
mv tempeval_output/wwi.ascii.TE3input.tml tempeval_output/news-annotated-output/
mv tempeval_output/james_brady_dies.ascii.TE3input.tml tempeval_output/news-annotated-output/
mv tempeval_output/chomsky_on_gaza.ascii.TE3input.tml tempeval_output/news-annotated-output/
mv tempeval_output/man_u_vs_liverpool.ascii.TE3input.tml       tempeval_output/news-annotated-output/
mv tempeval_output/russian_pilgrims.ascii.TE3input.tml tempeval_output/news-annotated-output/
mv tempeval_output/ebola_infection.ascii.TE3input.tml   tempeval_output/news-annotated-output/
mv tempeval_output/marijuana.ascii.TE3input.tml tempeval_output/news-annotated-output/
mv tempeval_output/microsoft_sues_samsung.ascii.TE3input.tml tempeval_output/news-annotated-output/

cp -r tempeval_output/wikipedia-annotated-output test-tools-and-data/qa-tempeval-test-wikipedia/systems/
cp -r tempeval_output/news-annotated-output test-tools-and-data/qa-tempeval-test-news/systems/
cp -r tempeval_output/blog-annotated-output test-tools-and-data/qa-tempeval-test-blogs/systems/

cd test-tools-and-data/qa-tempeval-test-wikipedia
bash evaluate_systems.sh

cd $TEA_PATH/test-tools-and-data/qa-tempeval-test-news
bash evaluate_systems.sh

cd $TEA_PATH/test-tools-and-data/qa-tempeval-test-blogs
bash evaluate_systems.sh




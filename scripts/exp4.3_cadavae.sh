bash infer_traditional.sh $1 0

if [ $1 = "CUB" ]; then
	bash infer_anymode_cub.sh standard interactive_siblings $1 0
	bash infer_anymode_cub.sh standard random $1 0
	bash infer_anymode_cub.sh standard interalllatent $1 0
fi
if [ $1 = "SUN" ]; then
	bash infer_anymode_sun.sh standard interactive_siblings $1 0
	bash infer_anymode_sun.sh standard random $1 0
	bash infer_anymode_sun.sh standard interalllatent $1 0
fi
if [ $1 = "AWA2" ]; then
	bash infer_anymode_awa2.sh standard interactive_siblings $1 0
	bash infer_anymode_awa2.sh standard random $1 0
	bash infer_anymode_awa2.sh standard interalllatent $1 0
fi


